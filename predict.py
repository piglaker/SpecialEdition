
import os
import random
import time
import logging
import argparse
from dataclasses import dataclass, field
from typing import Optional

import nltk
import numpy as np
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
    Seq2SeqTrainingArguments,
    set_seed,
)
from transformers import Trainer, Seq2SeqTrainer
from transformers import TrainingArguments
from transformers import trainer_utils, training_args
from transformers import BertForMaskedLM

import core
from core import get_dataset, get_metrics, argument_init 
from lib import subTrainer 
from data.DatasetLoadingHelper import load_ctc2021, load_sighan, load_sighan14_test, load_sighan15_test, load_magic_sighan
from models.bart.modeling_bart_v2 import BartForConditionalGeneration
from utils import levenshtein
from utils.io import read_csv, write_to
from bert_MaskedLM_v2 import MyDataCollatorForSeq2Seq, MyTrainer

class mydataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

def postprocess(preds, src):
    """
    since bertMaskedLM output "a,x,b,c,noise,noise", we truncate them
    """
    res = []
    for i in range(len(src)):
        res.append("".join(preds[i].split())[:len(src[i]["input_ids"])-2])
    
    return res

def get_dataset15():
    eval_data = load_sighan15_test()

    eval_dataset = mydataset(eval_data)

    return eval_dataset
    

def get_dataset14():

    eval_data = load_sighan14_test()

    eval_dataset = mydataset(eval_data)

    return eval_dataset

def predict_MaskLM(name, model, dataset, tokenizer, data_collator, compute_metrics):
    """
    
    """
    trainer = MyTrainer(
        model=model, 
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    predict_results = trainer.predict(
        dataset, 
    )

    predictions = np.where(predict_results.predictions != -100, predict_results.predictions, tokenizer.pad_token_id)

    metrics = predict_results.metrics

    print(metrics)

    predictions = tokenizer.batch_decode(
        sequences=predictions, 
        skip_special_tokens=True, 
        clean_up_tokenization_spaces=True
    )

    final = postprocess(predictions, dataset)

    predictions = [pred.strip() for pred in final]

    output_prediction_file =  "./predict/" + name + "generated_predictions"

    with open(output_prediction_file, "w") as writer:
        writer.write("\n".join(predictions))

    return

def predict_Seq2Seq(training_args, name, model, dataset, tokenizer, data_collator, compute_metrics):
    """
    
    """
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    predict_results = trainer.predict(
        dataset,
        max_length=128, 
        num_beams=4,
    )

    predictions = np.where(predict_results.predictions != -100, predict_results.predictions, tokenizer.pad_token_id)

    metrics = predict_results.metrics

    print(metrics)

    predictions = tokenizer.batch_decode(
        sequences=predictions, 
        skip_special_tokens=True, 
        clean_up_tokenization_spaces=True
    )

    final = postprocess(predictions, dataset)

    predictions = [pred.strip() for pred in final]

    output_prediction_file =  "./predict/" + name + "generated_predictions"

    with open(output_prediction_file, "w") as writer:
        writer.write("\n".join(predictions))

    return

def run_MaskLM(model, model_path):

    # Tokenizer   
    tokenizer_model_name_path="hfl/chinese-roberta-wwm-ext"

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_model_name_path
    )

    # Model
    #model_path = './tmp/bart_sighan_seq_eval_10epoch_bs64/checkpoint-5560'
    #model = BartForConditionalGeneration.from_pretrained(model_path)

    #model_path = "./tmp/sighan/bert_MaskedLM_eval.epoch10.bs48/checkpoint-19740"
    #model = BertForMaskedLM.from_pretrained(model_path)
 
    #data_collator

    data_collator = MyDataCollatorForSeq2Seq(
        #tokenizer=tokenizer,
        #model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=64
    )

    #metrics
    compute_metrics = get_metrics(tokenizer)

    #dataset
    
    dataset14 = get_dataset14()
    name = model_path.split("/")[-2] + model_path.split("/")[-1] + "_14_"
    predict_MaskLM(name, model, dataset14, tokenizer, data_collator, compute_metrics)    
    
    dataset15 = get_dataset15()
    name = model_path.split("/")[-2] + model_path.split("/")[-1] + "_15_"
    predict_MaskLM(name, model, dataset15, tokenizer, data_collator, compute_metrics) 

    return 

def run_Seq2Seq(model, model_path):
    
    training_args = argument_init(Seq2SeqTrainingArguments)

    # Tokenizer   
    tokenizer_model_name_path="hfl/chinese-roberta-wwm-ext"

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_model_name_path
    )

    # Model
    #model_path = './tmp/bart_sighan_seq_eval_10epoch_bs64/checkpoint-5560'
    #model = BartForConditionalGeneration.from_pretrained(model_path)

    #model_path = "./tmp/sighan/bert_MaskedLM_eval.epoch10.bs48/checkpoint-19740"
    #model = BertForMaskedLM.from_pretrained(model_path)
 
    #data_collator

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=64
    )

    #metrics
    compute_metrics = get_metrics(tokenizer)

    #dataset
    
    dataset14 = get_dataset14()
    name = model_path.split("/")[-2] + model_path.split("/")[-1] + "_14_"
    predict_Seq2Seq()(name, model, dataset14, tokenizer, data_collator, compute_metrics)    
    
    dataset15 = get_dataset15()
    name = model_path.split("/")[-2] + model_path.split("/")[-1] + "_15_"
    predict_Seq2Seq(training_args, name, model, dataset15, tokenizer, data_collator, compute_metrics) 

    return 

if __name__ == "__main__":
    from bert_MaskedLM import MyDataCollatorForSeq2Seq, MyTrainer
    model_path = "./tmp/sighan/bert_MaskedLM_eval.epoch10.bs48/checkpoint-19740"
    model = BertForMaskedLM.from_pretrained(model_path)

    run_MaskLM(model, model_path)

    #model_path = "./tmp/sighan/bart_Seq2Seq_eval.epoch30.bs96"
    #model = BartForConditionalGeneration.from_pretrained(model_path)    

    #run_Seq2Seq(model, model_path)
