import os
import random
import time
import logging
import argparse
from dataclasses import dataclass, field
from typing import Optional,Dict, Union, Any, Tuple, List

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
from transformers import Trainer, Seq2SeqTrainer
from transformers import TrainingArguments
from transformers import trainer_utils, training_args
from transformers.trainer_pt_utils import nested_detach
from transformers import BertForMaskedLM
from transformers.file_utils import PaddingStrategy
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import BatchEncoding, PreTrainedTokenizerBase
from transformers.training_args import TrainingArguments

from core import (
    get_model,
    get_dataset, 
    #get_metrics, 
    argument_init, 
    _get_ReaLiSe_dataset,
    MySeq2SeqTrainingArguments, 
)
from lib import MyTrainer, FoolDataCollatorForSeq2Seq 
from data.DatasetLoadingHelper import load_ctc2021, load_sighan
#from models.bart.modeling_bart_v2 import BartForConditionalGeneration
from transformers import BertForMaskedLM
from models.bert.modeling_bert_v3 import BertModelForCSC, BetterBertModelForCSC

#os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"

logger = logging.getLogger(__name__)

def get_metrics():

    import numpy as np
    from datasets import load_metric

    def compute_metrics(eval_preds):
        """
        reference: https://github.com/ACL2020SpellGCN/SpellGCN/blob/master/run_spellgcn.py
        """
        Achilles = time.time()

        sources, preds, labels = eval_preds# (num, length) np.array
 
        tp, fp, fn = 0, 0, 0

        sent_p, sent_n = 0, 0

        for i in range(len(sources)):
            print(sources[i])
            print(preds[i])
            print(labels[i])

            source, pred, label = sources[i], preds[i], labels[i]

            source, label = source[ source != -100], label[label != -100]

            source, label = source[source != 0],  label[label != 0]#pad idx for input_ids 

            #we guess pretrain Masked Language Model bert lack the surpvised sighan for 101 & 102 ( [CLS] & [SEP] ) , so we just ignore
            source, pred, label = np.where(source == 102, 101, source), np.where(pred == 102, 101, pred), np.where(label == 102, 101, label) 

            source, pred, label = source[ source != 101 ], pred[ pred != 101 ], label[ label != 101]

            source = source[:len(label)]
            pred = pred[:len(label)]

            pred = np.concatenate((pred, np.array([ 0 for i in range(len(label) - len(pred))])), axis=0)

            if len(pred) != len(source) or len(label) != len(source):
                print("Warning : something goes wrong when compute metrics, check codes now.")
                print(len(source), len(pred), len(label))
                print("source: ", source)
                print("pred: ", pred)
                print("label:", label)
                print("raw source: ", sources[i])
                print("raw pred: ", preds[i])
                print("raw label:", labels[i])
                exit()
            try:
                (pred != source).any()
            except:
                print(pred, source)
                print(" Error Exit ")
                exit(0)

            #if i < 5:
            print(source)
            print(pred)
            print(label)
            print((pred != source).any())
            print((pred == label).all())
            print((label != source).any())
            
            if (pred != source).any():
                sent_p += 1
                print("sent_p")
                if (pred == label).all():
                    tp += 1
                    print("tp")

            if (label != source).any():
                sent_n += 1
                print("sent_n")
            
        print(tp, sent_p, sent_n)

        precision = tp / (sent_p + 1e-10)

        recall = tp / (sent_n + 1e-10)

        F1_score = 2 * precision * recall / (precision + recall + 1e-10)

        Turtle = time.time() - Achilles

        if F1_score < 0.05:
            print("Warning : metric score is too Low , maybe something goes wrong, check your codes please.")
            #exit(0)
        return {"F1_score": float(F1_score), "Precision":float(precision),  "Recall":float(recall),"Metric_time":Turtle}

    return compute_metrics


def run():
    # Args
    training_args = argument_init(MySeq2SeqTrainingArguments)

    set_seed(training_args.seed)

    pretrained_csc_model = None#"/remote-home/xtzhang/CTC/CTC2021/SE_tmp_back/milestone/ReaLiSe/pretrained"#None

    # Tokenizer    
    tokenizer_model_name_path="hfl/chinese-roberta-wwm-ext" if pretrained_csc_model is None else pretrained_csc_model 

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_model_name_path 
    )

    # Dataset
    train_dataset, eval_dataset, test_dataset = _get_ReaLiSe_dataset(training_args.eval_dataset)#get_dataset(training_args.dataset) 

    # Model
    
    model = BertForMaskedLM.from_pretrained("hfl/chinese-roberta-wwm-ext")

    # Metrics
    compute_metrics = get_metrics()

    # Data Collator

    data_collator = FoolDataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=64
    )#my data collator  fix the length for bert.
    
    # Trainer
    trainer = MyTrainer(
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
