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

import sys
sys.path.append("/remote-home/xtzhang/CTC/CTC2021/SpecialEdition/")
from core import get_metrics, argument_init, get_dataset
from lib import subTrainer  
from models.bert.modeling_bert_v3 import BertForMaskedLM_v2 


from bert_MaskedLM import MyDataCollatorForSeq2Seq

metric = get_metrics()

train_dataset, eval_dataset, test_dataset= get_dataset("sighan", "../") 

tokenizer_model_name_path="hfl/chinese-roberta-wwm-ext"

tokenizer = AutoTokenizer.from_pretrained(
    tokenizer_model_name_path
)

preprocess = MyDataCollatorForSeq2Seq(tokenizer)

model = BertForMaskedLM.from_pretrained("/remote-home/xtzhang/CTC/CTC2021/SpecialEdition/tmp/sighan/bert_MaskedLM_base_raw.epoch10.bs128")

logits = model(**preprocess([test_dataset[0]])).logits

p = torch.softmax(logits, dim=2)

pre = tokenizer.batch_decode(torch.argmax(p, dim=2))

p_top5 = p.topk(5, dim=2)[-1]

p_top5 = p_top5.T.reshape(5, -1)

for i in p_top5:
    pres = tokenizer.convert_ids_to_tokens(i)
    print(pres)

bs = 2

result = []

print(tokenizer.convert_tokens_to_ids(["[CLS]", "[SEP]", "[EOS]"] ) )

exit()

for i in tqdm(range(0, len(test_dataset) // bs + 1)):
    batch = test_dataset[ i*32 : (i+1) *32]

    logits = model(**preprocess(batch)).logits

    #pred = torch.argmax(torch.softmax(logits, 2), -1)
    topks = []
    for i in logits:
        #print(i, i.shape)
        #exit()
        topks.append( torch.softmax(i, dim=1).topk(5, dim=1)[-1].T.reshape(5, -1) )
    
    print(topks)
    exit()
    

    result += mid
    break
print(result[:5])


