
import os
import random
import time
import logging
import argparse
from dataclasses import dataclass, field
from typing import Optional

import nltk
import numpy as np
import datasets
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import transformers
from transformers import (
    DataCollatorForSeq2Seq,
    AutoConfig,
    AutoTokenizer,
    HfArgumentParser,
    Seq2SeqTrainingArguments,
    set_seed,
)
from transformers import Trainer, Seq2SeqTrainer
from transformers import TrainingArguments
from transformers import trainer_utils, training_args

from core import (
    get_model,
    get_dataset, 
    get_seq2seq_metrics, 
    argument_init,
    MySeq2SeqTrainingArguments,
)
from lib import subTrainer 
from data.DatasetLoadingHelper import load_ctc2021, load_sighan
from models.bart.modeling_bart_v2 import BartForConditionalGeneration

#os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"

logger = logging.getLogger(__name__)


def run():
    # Args
    training_args = argument_init(MySeq2SeqTrainingArguments)

    set_seed(training_args.seed)

    # Tokenizer    
    tokenizer_model_name_path="hfl/chinese-roberta-wwm-ext"

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_model_name_path
    )

    # Dataset
    train_dataset, eval_dataset, test_dataset = get_dataset(training_args.dataset) 

    # Model
    model = get_model(
        model_name =  "CPT_NLG" if training_args.model_name is None else training_args.model_name, 
        #'/remote-home/share/yfshao/bart-zh/arch12-2-new-iter8w/'
        pretrained_model_name_or_path='/remote-home/xtzhang/CTC/CTC2021/SpecialEdition/models/bart/bart-zh/arch12-2-new-iter8w'
    ) #base

    # Metrics
    compute_metrics = get_seq2seq_metrics()

    # Data Collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=None
    )
    
    # Trainer
    trainer = subTrainer(
        model=model,
        args=training_args,         
        train_dataset=train_dataset,    
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,      
        compute_metrics=compute_metrics#hint:num_beams and max_length effect heavily on metric["F1_score"], so I modify train_seq2seq.py to value default prediction_step function
    )

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

    max_length = 128
    
    num_beams =  4

    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        
        metrics = trainer.evaluate(
            max_length=training_args.max_length, 
            num_beams=training_args.num_beams, 
            metric_key_prefix="eval"
            )
        
        metrics["eval_samples"] = len(eval_dataset)

        trainer.log_metrics("eval", metrics)
        
        trainer.save_metrics("eval", metrics)

    # Predict
    if training_args.do_predict:
        logger.info("*** Predict ***")

        predict_results = trainer.predict(
            test_dataset, 
            metric_key_prefix="predict", 
            max_length=max_length, 
            num_beams=num_beams
        )

        metrics = predict_results.metrics

        metrics["predict_samples"] = len(test_dataset)
        
        predictions = tokenizer.batch_decode(
                predict_results.predictions, 
                skip_special_tokens=True, 
                clean_up_tokenization_spaces=True
        )

        predictions = [pred.strip() for pred in predictions]

        output_prediction_file = os.path.join(training_args.output_dir, "generated_predictions.txt")

        with open(output_prediction_file, "w") as writer:
            writer.write("\n".join(predictions))

    logger.info("*"*10 + "Curtain" + "*"*10)

    return


if __name__ == "__main__":
    run()
