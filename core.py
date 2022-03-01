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

from data.DatasetLoadingHelper import (
    load_ctc2021, 
    load_sighan, 
    load_sighan_gector,
    load_lattice_sighan, 
    load_abs_pos_sighan, 
    load_abs_pos_sighan_lang8, 
    load_abs_pos_sighan_plus, 
    load_abs_pos_and_spe_token_sighan,
    load_sighan13_test,
    load_sighan14_test,
    load_sighan15_test,
)

@dataclass
class MySeq2SeqTrainingArguments(Seq2SeqTrainingArguments):
    model_name: str=field(default="MaskedLM", metadata={"help":"which bert model "})
    dataset: str = field(default="sighan", metadata={"help":"dataset"})
    eval_dataset:str = field(default="sighan", metadata={"help":"dataset for eval"})
    max_length: int = field(default=128, metadata={"help": "max length"})
    num_beams: int = field(default=4, metadata={"help": "num beams"})
    use_extra_dataset:bool = field(default=False, metadata={"help":"Only work for ctc2021, using larget v2"})


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


def get_model(model_name="MaskedLM", pretrained_model_name_or_path="hfl/chinese-roberta-wwm-ext"):
    """
    Just get model
    MLP:
        bert->mlp->loss
    Dot:
        bert->dot product with embeddings->loss
    MaskedLM_v2:
        lexcions ( flat
    CL:
        Model with Contrastive Learning Loss
    MaskedLM:
        bert->lmhead->loss
    """
    model = None

    print("Hint: Loading Model " + "*"*5 + model_name + "*"*5)

    if model_name == "MLP":
        from models.bert.modeling_bert_v3 import BertModelForCSC as ProtoModel
    elif model_name == "Dot":
        from models.bert.modeling_bert_v3 import BetterBertModelForCSC as ProtoModel
    elif model_name == "MaskedLM_v2":
        from models.bert.modeling_bert_v3 import BertForMaskedLM_v2  as ProtoModel
    elif model_name == "CL":
        from models.bert.modeling_bert_v4 import BertForMaskedLM_CL as ProtoModel
    elif model_name == "CPT_NLG":
        from models.bart.modeling_bart_v2 import BartForConditionalGeneration as ProtoModel
        pretrained_model_name_or_path="fnlp/cpt-base" # '/remote-home/xtzhang/CTC/CTC2021/SpecialEdition/models/bart/bart-zh/arch12-2-new-iter8w'
        #pretrained_model_name_or_path = '/remote-home/xtzhang/CTC/CTC2021/SpecialEdition/models/bart/bart-zh/arch12-2-new-iter8w'
    elif model_name == "CPT_NLU":
        from models.bart.modeling_bart_v2 import BartForMaskedLM as ProtoModel
        pretrained_model_name_or_path="fnlp/cpt-large" # '/remote-home/xtzhang/CTC/CTC2021/SpecialEdition/models/bart/bart-zh/arch12-2-new-iter8w'
    elif model_name == "BART-base":
        from models.bart.modeling_bart_v2 import BartForConditionalGeneration as ProtoModel
        pretrained_model_name_or_path="fnlp/bart-base-chinese"# '/remote-home/xtzhang/CTC/CTC2021/SpecialEdition/models/bart/bart-zh/arch12-2-new-iter8w'
    elif model_name == "BART-large":
        from models.bart.modeling_bart_v2 import BartForConditionalGeneration as ProtoModel
        pretrained_model_name_or_path="fnlp/bart-large-chinese"# '/remote-home/xtzhang/CTC/CTC2021/SpecialEdition/models/bart/bart-zh/arch12-2-new-iter8w'
    elif model_name == "Proto":
        from models.bert.modeling_bert_v4 import ProtoBertForMaskedLM as ProtoModel
    elif model_name == "Gector":
        from models.bert.modeling_bert_v3 import GectorModel as ProtoModel
    elif model_name is None or model_name == "MaskedLM":
        print("Hint: Load Default BertForMaskedLM.")
        from transformers import BertForMaskedLM as ProtoModel
    else:
        print("Hint: No such " + model_name)
        exit(0)

    model = ProtoModel.from_pretrained(pretrained_model_name_or_path=pretrained_model_name_or_path)

    if not model:
        print("Warning: wrong model name ! You ")
        exit()
    return model


def get_dataset(training_args):
    """
    preprocess wrapped in load_ctc2021
    return : mydate
                torch.LongTensor
    
        Good day!
    """

    print("Loading Dataset !")
    os.system("date")

    if training_args.dataset == "ctc2021":
        train_data, eval_data, test_data = load_ctc2021(training_args.use_extra_dataset)
    elif training_args.dataset == "sighan":
        train_data, eval_data, test_data = load_sighan(path_head="")
    else:
        print("Error: No such dataset ")
        print(training_args.dataset)
        exit(0)

    train_dataset, eval_dataset, test_dataset = mydataset(train_data), mydataset(eval_data), mydataset(test_data)

    print("Loading Succeed !")
    os.system("date")

    return train_dataset, eval_dataset, test_dataset


def get_dataset_plus(training_args):
    """
    preprocess wrapped in load_ctc2021
    return : mydate
                torch.LongTensor
    
        Good day!
    """

    print("Loading Dataset !")
    os.system("date")

    if training_args.dataset == "ctc2021":
        train_data, eval_data, test_data = load_ctc2021(extra=training_args.use_extra_dataset)
    elif training_args.dataset == "sighan":
        #train_data, eval_data, test_data = load_sighan(path_head)
        if training_args.model_name == "Gector":
            return get_Gector_dataset()
        else:
            return get_ReaLiSe_dataset()
    else:
        print("Error: No such dataset ")
        print(training_args.dataset)
        exit(0)

    train_dataset, eval_dataset, test_dataset = mydataset(train_data), mydataset(eval_data), mydataset(test_data)

    print("Loading Succeed !")
    os.system("date")

    return train_dataset, eval_dataset, test_dataset


def get_Gector_dataset(which="15"):
    """
    Gector for sighan
    """
    print("Loading Dataset !")
    os.system("date")

    train_data, eval_data, test_data = load_sighan_gector(path_head="")
    
    train_dataset, eval_dataset, test_dataset = mydataset(train_data), mydataset(eval_data), mydataset(test_data)

    print("Loading Succeed !")
    os.system("date")

    return train_dataset, eval_dataset, test_dataset


def get_ReaLiSe_dataset(which="15"):
    """
    For its 
    """
    print("Loading Dataset !")
    print("Hint: The Data You loading now is the preprocessed sighan from ReaLise, ")
    os.system("date")

    path = "../SE_tmp_back/milestone/ReaLiSe/data/"
    import pickle
    train_dataset = pickle.load(open(path + "trainall.times2.pkl", "rb"))
    eval_dataset = pickle.load(open(path + "test.sighan" + which + ".pkl", "rb"))
    test_dataset = pickle.load(open(path + "test.sighan" + which + ".pkl", "rb"))

    print("Hint: Using **SIGHAN" + which + "** for eval & test !")

    def trans2mydataset(features):
        new = []
        for feature in features:
            tmp = {}
            tmp["input_ids"] = feature["src_idx"][:128]
            tmp["labels"] = feature["tgt_idx"][:128]
            tmp["attention_mask"] = ([1] * len(tmp["input_ids"]))[:128]#feature["lengths"])[:128]
            new.append(tmp)
        
        return mydataset(new)

    print("Loading Succeed !")
    os.system("date")

    return trans2mydataset(train_dataset), trans2mydataset(eval_dataset), trans2mydataset(test_dataset)


def get_sighan_test(which, path_head=""):
    """
    """
    print("Loading Dataset !")
    os.system("date")
    if which == "13":
        test_data = load_sighan13_test(path_head)
    elif which == "14":
        test_data = load_sighan14_test(path_head)
    elif which == "15":
        test_data = load_sighan15_test(path_head)
    else:
        print("Error: No such dataset ")
        print(which)
        exit(0)

    test_dataset = mydataset(test_data)

    print("Loading Succeed !")
    os.system("date")

    return test_dataset


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


def get_magic_plus_dataset(dataset="sighan", path_head=""):
    """
    """
    print("Loading Dataset !")
    os.system("date")

    if dataset == "sighan":
        train_data, eval_data, test_data = load_abs_pos_sighan_plus(path_head=path_head)
    else:
        exit()

    train_dataset, eval_dataset, test_dataset = mydataset(train_data), mydataset(eval_data), mydataset(test_data)

    print("Loading Succeed !")

    os.system("date")

    return train_dataset, eval_dataset, test_dataset


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


def get_magic_lang8_dataset(dataset="sighan", path_head=""):
    """
    """
    print("Loading Dataset !")
    os.system("date")

    if dataset == "sighan":
        train_data, eval_data, test_data = load_abs_pos_sighan_lang8(path_head=path_head)
    else:
        exit()

    train_dataset, eval_dataset, test_dataset = mydataset(train_data), mydataset(eval_data), mydataset(test_data)

    print("Loading Succeed !")

    os.system("date")

    return train_dataset, eval_dataset, test_dataset


def get_magic_expand_dataset(dataset="sighan", path_head=""):
    """
    """
    print("Loading Dataset !")
    os.system("date")

    if dataset == "sighan":
        train_data, eval_data, test_data = load_abs_pos_sighan_plus(path_head=path_head)
    else:
        exit()

    train_dataset, eval_dataset, test_dataset = mydataset(train_data), mydataset(eval_data), mydataset(test_data)

    print("Loading Succeed !")

    os.system("date")

    return train_dataset, eval_dataset, test_dataset


def get_super_magic_dataset(dataset="sighan", path_head=""):
    """
    """
    print("Loading Dataset !")
    os.system("date")

    if dataset == "sighan":
        train_data, eval_data, test_data, tokenizer = load_abs_pos_and_spe_token_sighan(path_head=path_head)
    else:
        exit()

    train_dataset, eval_dataset, test_dataset = mydataset(train_data), mydataset(eval_data), mydataset(test_data)

    print("Loading Succeed !")

    os.system("date")

    return train_dataset, eval_dataset, test_dataset, tokenizer


def get_metrics(training_args):
    if training_args.dataset == "sighan":
        return _get_metrics(training_args)
    if training_args.dataset == "ctc2021":
        return _get_seq2seq_metrics(training_args)
    else:
        print("Error when getting metrics.")
        exit(0)


def _get_metrics(training_args):
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
            source, pred, label = sources[i], preds[i], labels[i]
            #print(source)
            #print(pred)
            #print(label) 
            #exit()
            source, label = source[ source != -100], label[label != -100]
            #source, pred, label = source[source != -100], pred[pred != -100], label[label != -100]# pad idx for labels
            # print(source)
            # print(pred)
            # print(label)
            #source, pred, label = source[source != 102], pred[ pred != 102 ], label[ label != 102]
            #source, pred, label = source[source != 101], pred[pred != 101], label[label != 101]#remove  
            source, label = source[source != 0],  label[label != 0]#pad idx for input_ids 
            #source, pred, label = source[source != -100], pred[pred != -100], label[label != -100] 

            source = source[:len(label)]
            pred = pred[:len(label)]
            #print(source, pred, label)    
            #print(type(pred), type(source), pred==source)

            #we guess pretrain Masked Language Model bert lack the surpvised sighan for 101 & 102 ( [CLS] & [SEP] ) , so we just ignore
            source, pred, label = np.where(source == 102, 101, source), np.where(pred == 102, 101, pred), np.where(label == 102, 101, label) 
            #source, pred, label = source[1:len(source)-1], pred[1:len(pred)-1], label[1:len(label)-1]

            # print("source", source)
            # print("pred", pred)
            # print("label",label)

            # print( (pred != source).any() )
            # print( (pred == label).all() )
            # print( (label != source).any() )

            if len(pred) != len(source) or len(label) != len(source):
                print("Warning : something goes wrong when compute metrics, check codes now.")
                print(len(source), len(pred), len(label))
                print("source: ", source)
                print("pred: ", pred)
                print("label:", label)
                exit()

            try:
                (pred != source).any()
            except:
                print(pred, source)
                print(" Error Exit ")
                exit(0)

            # print((pred != 1).any())
            # print((pred == label).all())
            # print((label != 1).any()) 
            
            if training_args.model_name != "Gector":
                # label: [101, 2,... 3, 102]
                if (pred != source).any():
                    sent_p += 1
                    if (pred == label).all():
                        tp += 1

                if (label != source).any():
                    sent_n += 1
            else:
                # label : [ 1,1,1,1,1 ]
                if (pred != 1).any():
                    sent_p += 1

                    if (pred == label).all():
                        tp += 1
            
                if (label != 1).any():
                    sent_n += 1

        precision = tp / (sent_p + 1e-10)

        recall = tp / (sent_n + 1e-10)

        F1_score = 2 * precision * recall / (precision + recall + 1e-10)

        Turtle = time.time() - Achilles

        if F1_score < 0.1:
            print("Warning : metric F1_score is too Low , maybe something goes wrong, check your codes please.")

        return {"F1_score": float(F1_score), "Precision":float(precision),  "Recall":float(recall),"Metric_time":Turtle}

    return compute_metrics


def _get_seq2seq_metrics(training_args):
    """
    Main difference from " get_metrics() " is that seq2seq output & labels doesn't match (length).
    #https://huggingface.co/metrics
    #accuracy,bertscore, bleu, bleurt, coval, gleu, glue, meteor,
    #rouge, sacrebleu, seqeval, squad, squad_v2, xnli
    metric = load_metric() 
    """
    
    import numpy as np
    from datasets import load_metric

    def compute_metrics(eval_preds):
        """
        reference: https://github.com/destwang/CTC2021

        >>> final_score = 0.8 * detect_f1 + 0.2 * correct_f1
        """
        Achilles = time.time()

        sources, preds, labels = eval_preds# (num, length) np.array
 
        tp_detect, tp_correct = 0, 0

        p, n = 0, 0

        for i in range(len(sources)):
            source, pred, label = sources[i], preds[i], labels[i]

            source, label = source[ source != -100], label[label != -100]
            source, label = source[source != 0],  label[label != 0]#pad idx for input_ids 
 
            # source = source[:len(label)]
            # pred = pred[:len(label)]
            pred = pred[ pred != 0]

            #we guess pretrain Masked Language Model bert lack the surpvised sighan for 101 & 102 ( [CLS] & [SEP] ) , so we just ignore
            source, pred, label = np.where(source == 102, 101, source), np.where(pred == 102, 101, pred), np.where(label == 102, 101, label) 

            from utils.levenshtein import levenshtein4seq
            
            # print(source.shape, pred.shape, label.shape)
            # print(source)
            # print(pred)
            # print(label)

            pred_edits, gold_edits = levenshtein4seq(source, pred, only_edits=True), levenshtein4seq(source, label, only_edits=True)

            # print(pred_edits)
            # print(gold_edits)

            # gold_pos = [ edit[1] for edit in gold_edits]

            p += len(pred_edits)

            n += len(gold_edits)

            ref_error_set = set(gold_edits)
            
            ref_det_set, ref_cor_set = set( [ (edit[0], edit[1]) for edit in gold_edits ] ), set( [ (edit[-1]) for edit in gold_edits] )

            pred_error_set = set(pred_edits)

            pred_det_set, pred_cor_set = set( [ (edit[0], edit[1]) for edit in pred_edits ] ), set( [(edit[-1]) for edit in pred_edits] ) 

            n += len(ref_cor_set)
            p += len(pred_cor_set)

            for error in ref_error_set:
                loc, typ, wrong, cor_text = error
                if (loc, wrong) in pred_det_set or (cor_text) in pred_cor_set:
                    tp_detect += 1

            tp_correct += len(ref_cor_set & pred_cor_set)

            # for edit in gold_edits:
                # if edit in pred_edits or edit in pred_cor_set:
                    # det_right_num += 1

            # for edit in pred_edits:
                # if edit[1] in gold_pos:
                    # tp_detect += 1
                # if edit in gold_edits:
                    # tp_correct += 1

        def compute_f1score(tp, p, n):

            precision = tp / (p + 1e-10)
            recall = tp / (n + 1e-10)
            F1_score = 2 * precision * recall / (precision + recall + 1e-10)

            return F1_score, precision, recall

        Turtle = time.time() - Achilles

        F1_score_detect, Precision_detect, Recall_detect = compute_f1score(tp_detect, p, n)
        F1_score_correct, Precision_correct, Recall_correct = compute_f1score(tp_correct, p, n)

        if F1_score_detect < 0.1 or F1_score_correct < 0.1:
            print("Warning : metric F1_score is too Low , maybe something goes wrong, check your codes please.")

        return {"F1_score":float( 0.8 * F1_score_detect + 0.2 * F1_score_correct ), 
                "F1_score_detect": float(F1_score_detect), 
                "Precision_detect":float(Precision_detect),  
                "Recall_detect":float(Recall_detect),
                "F1_score_correct": float(F1_score_correct), 
                "Precision_correct":float(Precision_correct),  
                "Recall_correct":float(Recall_correct),
                "Metric_time":Turtle}

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