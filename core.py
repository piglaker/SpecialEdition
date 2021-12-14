import os
import time
from dataclasses import dataclass, field
from typing import Optional,Dict, Union, Any, Tuple, List

import nltk
import numpy as np
import datasets
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import transformers
from transformers import (
    DataCollatorForLanguageModeling,
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

from data.DatasetLoadingHelper import load_ctc2021, load_sighan, load_lattice_sighan, load_abs_pos_sighan


class mydataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


def argument_init(trainingarguments=Seq2SeqTrainingArguments):

    """
    
    """
    parser = HfArgumentParser(trainingarguments)

    training_args = parser.parse_args_into_dataclasses()[0]

    return training_args


def get_dataset(dataset, path_head=""):
    """
    preprocess wrapped in load_ctc2021
    return : mydate
                torch.LongTensor
    
        Good day!
    """

    print("Loading Dataset !")
    os.system("date")

    if dataset == "ctc2021":
        train_data, eval_data, test_data = load_ctc2021()
    elif dataset == "sighan":
        train_data, eval_data, test_data = load_sighan(path_head)
    else:
        print("Error: No such dataset ")
        print(dataset)
        exit(0)

    train_dataset, eval_dataset, test_dataset = mydataset(train_data), mydataset(eval_data), mydataset(test_data)

    print("Loading Succeed !")
    os.system("date")

    return train_dataset, eval_dataset, test_dataset


def get_lattice_dataset(dataset="sighan", path_head="."):
    """
    """
    print("Loading Dataset !")
    os.system("date")

    if dataset == "sighan":
        datasets, vocabs, embeddings = load_lattice_sighan(path_head=path_head)
    else:
        exit()

    datasets["train"], datasets["valid"], datasets["test"] = mydataset(datasets["train"]), mydataset(datasets["valid"]), mydataset(datasets["test"])

    return datasets, vocabs, embeddings


def get_magic_dataset(dataset="sighan", path_head=""):
    """
    """
    print("Loading Dataset !")
    os.system("date")

    if dataset == "sighan":
        train_data, eval_data, test_data = load_abs_pos_sighan(path_head=path_head)
    else:
        exit()

    train_dataset, eval_dataset, test_dataset = mydataset(train_data), mydataset(eval_data), mydataset(test_data)

    print("Loading Succeed !")

    os.system("date")

    return train_dataset, eval_dataset, test_dataset


def get_metrics():
    """
    #https://huggingface.co/metrics
    #accuracy,bertscore, bleu, bleurt, coval, gleu, glue, meteor,
    #rouge, sacrebleu, seqeval, squad, squad_v2, xnli
    metric = load_metric() 
    """
    
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
            #print(sources[i])
            #print(preds[i])
            #print(labels[i])

            source, pred, label = sources[i][sources[i] != 102], preds[i][ preds[i] != 102 ], labels[i][ labels[i] != 102]
            source, pred, label = source[source != 101], pred[pred != 101], label[label != 101]
            source, pred, label = source[source != 0], pred[pred != 0], label[label != 0]
            source, pred, label = source[source != -100], pred[pred != -100], label[label != -100] 

            source = source[:len(label)]
            pred = pred[:len(label)]        

            if (pred != source).any():
                sent_p += 1
                if (pred == label).all():
                    tp += 1

            if (label != source).any():
                sent_n += 1

        precision = tp / (sent_p + 1e-10)

        recall = tp / (sent_n + 1e-10)

        F1_score = 2 * precision * recall / (precision + recall + 1e-10)

        Turtle = time.time() - Achilles

        return {"F1_score": float(F1_score), "Precision":float(precision),  "Recall":float(recall),"Metric_time":Turtle}

    return compute_metrics


def get_lattice_metrics():
    """
    """
    
    import numpy as np
    from datasets import load_metric

    def compute_metrics(eval_preds):
        """
        """
        Achilles = time.time()

        sources, preds, labels = eval_preds# (num, length) np.array

        tp, fp, fn = 0, 0, 0

        sent_p, sent_n = 0, 0

        for i in range(len(sources)):
            
            source, pred, label = sources[i], preds[i], labels[i]
            source, pred, label = source[source != -100], pred[pred != -100], label[label != -100] 

            source = source[:len(label)]
            pred = pred[:len(label)]

            if (pred != source).any():
                sent_p += 1
                if (pred == label).all():
                    tp += 1

            if (label != source).any():
                sent_n += 1

        precision = tp / sent_p
        recall = tp / sent_n
        F1_score = 2 * precision * recall / (precision + recall + 1e-10)

        Turtle = time.time() - Achilles

        return {"F1_score": float(F1_score), "Precision":float(precision),  "Recall":float(recall),"Metric_time":Turtle}

    return compute_metrics


if __name__ == "__main__":
    print("Lets test !")

    training_args = argument_init(TrainingArguments)

    train_dataset, eval_dataset, test_dataset = get_dataset(training_args.dataset) 

    compute_metrics = get_metrics(None)

    print("Done")