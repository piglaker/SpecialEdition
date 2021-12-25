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
import torch.functional as F
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


def top_k_top_p_filtering(logits, top_k=0, top_p=1.0, filter_value=-float("Inf"), min_tokens_to_keep=2):
    """
    根据top_k, top_p的值，将不满足的值置为filter_value的值

    :param torch.Tensor logits: bsz x vocab_size
    :param int top_k: 如果大于0，则只保留最top_k的词汇的概率，剩下的位置被置为filter_value
    :param int top_p: 根据(http://arxiv.org/abs/1904.09751)设置的筛选方式
    :param float filter_value:
    :param int min_tokens_to_keep: 每个sample返回的分布中有概率的词不会低于这个值
    :return:
    """
    if top_k > 0:
        top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        print(logits, indices_to_remove)
        logits[indices_to_remove] = filter_value

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs > top_p
        if min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
            sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    return logits


metric = get_metrics()

train_dataset, eval_dataset, test_dataset= get_dataset("sighan", "../") 

tokenizer_model_name_path="hfl/chinese-roberta-wwm-ext"

tokenizer = AutoTokenizer.from_pretrained(
    tokenizer_model_name_path
)

preprocess = MyDataCollatorForSeq2Seq(tokenizer)

model = BertForMaskedLM.from_pretrained("/remote-home/xtzhang/CTC/CTC2021/SE_tmp_back/tmp/sighan/bert_MaskedLM_base_raw.epoch10.bs128")

logits = model(**preprocess([test_dataset[0]])).logits

p = torch.softmax(logits, dim=2)

pre = tokenizer.batch_decode(torch.argmax(p, dim=2))

p_top5 = p.topk(5, dim=2)[-1]

p_top5 = p_top5.T.reshape(5, -1)

bs = 32

result = []

#print(tokenizer.convert_tokens_to_ids(["[CLS]", "[SEP]", "[EOS]"] ) )

#exit()

tp, sent_p, sent_n = 0, 0, 0

tp2 = 0

for i in tqdm(range(0, len(test_dataset) // bs + 1)):
    batch = test_dataset[ i*bs : (i+1) *bs]

    logits = model(**preprocess(batch)).logits

    #pred = torch.argmax(torch.softmax(logits, 2), -1)
    #print(logits.shape)
    topks = []

    for i in logits:
        #topks.append( torch.softmax(i, dim=1).topk(5, dim=1)[-1].T.reshape(5, -1) )
        topks.append(top_k_top_p_filtering(i, top_k=5))
        print(topks)
        exit()
    labels = torch.tensor([i["labels"] for i in batch])
    labels = [ i[ i != -100] for i in labels ]
    #labels = [ i[ i != -101] for i in labels ]
    #labels = [ i[ i != -102] for i in labels ]
    labels = [ i[ i != 0] for i in labels ]
    
    cut_topks = []
    for j in range(len(batch)):

        src, top5, label = torch.tensor(batch[j]["input_ids"]), topks[j], labels[j]

        src = src[ src != -100]
        src = src[ src != 0]

        #print(top5.shape, label.shape)
        
        tmp = top5[:, :label.shape[0]]
        
        #print(tmp.shape, tmp.T.reshape(-1,5) == label.view(-1, 1))
        
        res = (tmp.T.reshape(-1, 5) == label.view(-1, 1)).sum().item()
        
        sent_p += 1
        if res == label.shape[0]:
            tp += 1

        if ( src == label).sum().item() != label.shape[0]:
            sent_n += 1
            if res == label.shape[0]:
                tp2 += 1

precision = tp / (sent_p + 1e-10)

recall = tp2 / (sent_n + 1e-10)

F1_score = 2 * precision * recall / (precision + recall + 1e-10)
    
print("pre: ", precision, "recall: ", recall, "F1: ", F1_score)




