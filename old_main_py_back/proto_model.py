# Copyright 2021 piglake
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import re
import random
import time
import logging
import argparse
from dataclasses import dataclass, field
from typing import Optional,Dict, Union, Any, Tuple, List
from joblib import parallel_backend

import fitlog
import numpy as np
import datasets
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import transformers
from transformers import (
    DataCollatorForSeq2Seq,
    AutoConfig,
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
    Seq2SeqTrainingArguments,
    set_seed,
)

from transformers import TrainingArguments
from transformers import trainer_utils, training_args
from transformers.trainer_pt_utils import nested_detach
from transformers import BertForMaskedLM
from transformers.file_utils import PaddingStrategy
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import BatchEncoding, PreTrainedTokenizerBase
from transformers.training_args import TrainingArguments

from core import (
    fitlogging,
    get_model,
    get_dataset, 
    get_metrics, 
    argument_init, 
    get_dataset_plus,
    _get_mask_dataset,
    MySeq2SeqTrainingArguments, 
)
from lib import MyTrainer, FoolDataCollatorForSeq2Seq, subTrainer 
from data.DatasetLoadingHelper import load_ctc2021, load_sighan
#from models.bart.modeling_bart_v2 import BartForConditionalGeneration
from transformers import BertForMaskedLM

#os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"

logger = logging.getLogger(__name__)

#fitlog.set_log_dir("./fitlogs/")
#fitlog.add_hyper_in_file(__file__)


def run():
    # Args
    training_args = argument_init(MySeq2SeqTrainingArguments)

    #fitlogging(training_args)
    
    set_seed(training_args.seed)

    pretrained_csc_model = "hfl/chinese-macbert-base"#"hfl/chinese-roberta-wwm-ext"#"bert-base-chinese"#None#"/remote-home/xtzhang/CTC/CTC2021/SE_tmp_back/milestone/ReaLiSe/pretrained"#None

    # Tokenizer    
    tokenizer_model_name_path="hfl/chinese-roberta-wwm-ext" if pretrained_csc_model is None else pretrained_csc_model 

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_model_name_path 
    )

    # Dataset
    train_dataset, eval_dataset, test_dataset = get_dataset_plus(training_args)#get_dataset(training_args.dataset) 
    # train_dataset, eval_dataset, test_dataset = _get_mask_dataset(training_args)

    # Model
    model = get_model(
        model_name= "Dot" if training_args.model_name is None else training_args.model_name, 
        pretrained_model_name_or_path="hfl/chinese-roberta-wwm-ext" if pretrained_csc_model is None else pretrained_csc_model,  #"bert-base-chinese" 
        training_args=training_args,
    ) #base

    # Fix cls
    #if training_args.fix_cls:    
    #    for name, param in model.named_parameters():
    #        if 'cls' in name:
    #            param.requires_grad = False

    # Metrics
    compute_metrics = get_metrics(training_args)

    # Data Collator

    data_collator = FoolDataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=64
    )#my data collator  fix the length for bert.
    
    # Trainer
    if training_args.model_name in [ "MLP", "CL", "Dot", "Proto", "MaskedLM_v2", "CPT_NLU", "Gector", "MaskedLM"]:
        Trainer = MyTrainer # MaskedLM        
    elif training_args.model_name in  [ "CPT_NLG", "BART-base", "BART-large" ] :
        Trainer = subTrainer # Seq2Seq
        training_args.predict_with_generate = True
    else :
        print(" Error: Unregistered Model !")
        exit(0)

    trainer = Trainer(
        model=model,
        args=training_args,         
        train_dataset=train_dataset,    
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,      
        compute_metrics=compute_metrics,#hint:num_beams and max_length effect heavily on metric["F1_score"], so I modify train_seq2seq.py to value default prediction_step function
    )

    # fitlog.finish()

    # Train
    train_result = trainer.train()

    trainer.save_model()  # Saves the tokenizer too for easy upload

    metrics = train_result.metrics
    
    metrics["train_samples"] = len(train_dataset)

    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    # Evaluation 
    # reference:https://github1s.com/huggingface/transformers/blob/master/examples/pytorch/summarization/run_summarization.py

    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        
        metrics = trainer.evaluate(metric_key_prefix="eval")
        
        metrics["eval_samples"] = len(eval_dataset)

        trainer.log_metrics("eval", metrics)
        
        trainer.save_metrics("eval", metrics)

    # Predict
    if training_args.do_predict:
        logger.info("*** Predict ***")

        predict_results = trainer.predict(
            test_dataset, 
            metric_key_prefix="predict",
        )

        predictions = np.where(predict_results.predictions != -100, predict_results.predictions, tokenizer.pad_token_id)#remove the pad 

        metrics = predict_results.metrics

        metrics["predict_samples"] = len(test_dataset)

        #print(torch.tensor(predictions).shape)

        predictions = tokenizer.batch_decode(
                sequences=predictions, 
                skip_special_tokens=True, 
                clean_up_tokenization_spaces=True
        )

        predictions = [pred.strip() for pred in predictions]

        output_prediction_file = os.path.join(training_args.output_dir, "generated_predictions.txt")

        with open(output_prediction_file, "w") as writer:
            writer.write("\n".join(predictions))

    logger.info("*"*10 + "Curtain" + "*"*10)

    print("*"*10 + "over" + "*"*10)

    return

if __name__ == "__main__":
    run()
