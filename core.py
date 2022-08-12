import os
import re
import time
from dataclasses import dataclass, field
from timeit import repeat
from typing import Optional,Dict,Union,Any,Tuple,List

import fitlog
import nltk
import numpy as np
import datasets
import torch
import torch.nn as nn
import torch.distributed as dist
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
    load_sighan_enchanted, 
    load_sighan_gector,
    load_sighan_mask,
    load_sighan_expand,
    load_lattice_sighan, 
    load_abs_pos_sighan, 
    load_abs_pos_sighan_lang8, 
    load_abs_pos_sighan_plus, 
    load_abs_pos_and_spe_token_sighan,
    load_sighan13_test,
    load_sighan14_test,
    load_sighan15_test,
    load_sighan_chinesebert,
    load_sighan_chinesebert_mask,
    load_sighan_chinesebert_holy, 
    load_sighan_holy,
    load_sighan_holy_mask,
    load_sighan_pure, 

)

def ddp_exec(command):
    """
    """
    if os.environ["LOCAL_RANK"] != '0':
        return
    else:
        exec(command) 

def ddp_print(*something):
    """
    out of time
    """
    if os.environ["LOCAL_RANK"] != '0':
        return
    else:
        for thing in something:
            print(thing)

        return 

def fitlogging(training_args):
    for attr in dir(training_args):
        if not re.match("__.*__", attr) and isinstance(getattr(training_args, attr), (int, str, bool, float)):
            fitlog.add_hyper(value=getattr(training_args, attr), name=attr)
    return

@dataclass
class MySeq2SeqTrainingArguments(Seq2SeqTrainingArguments):
    # hack for bug
    disable_tqdm: Optional[bool] = field(
        default=None, metadata={"help": "Whether or not to disable the tqdm progress bars."}
    )

    load_best_model_at_end: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether or not to load the best model found during training at the end of training."},
    )

    # extra args
    model_name: str=field(default="MaskedLM", metadata={"help":"which bert model "})
    dataset: str = field(default="sighan", metadata={"help":"dataset"})
    eval_dataset:str = field(default="sighan", metadata={"help":"dataset for eval"})
    max_length: int = field(default=128, metadata={"help": "max length"})
    num_beams: int = field(default=4, metadata={"help": "num beams"})
    use_extra_dataset:bool = field(default=False, metadata={"help":"Only work for ctc2021, using larger v2"})
    fix_cls:bool = field(default=False, metadata={"help":"whether or not fix the cls layer of BertMaskedLM"})
    cl_weight:float = field(default=0.2, metadata={"help": "contrastive learning loss weight"})
    repeat_weight:float = field(default=0.2, metadata={"help": "distill repeat loss"})
    copy_weight:float = field(default=0.5, metadata={"help":"copy weight"})
    num_gpus:int = field(default=4, metadata={"help":"num_gpus"})
    pretrained_name:str = field(default="roberta", metadata={"help":"pretrained_name"})
    log_path:str = field(default="Recent_train.log", metadata={"help":"log path or name"})

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


def get_model(model_name="MaskedLM", pretrained_model_name_or_path="hfl/chinese-roberta-wwm-ext", training_args=None):
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
    T5:
        generation
    Proto:
        Prototype
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
    elif model_name == "T5-base":
        print("Warning: T5-base is not implemented yet")
        exit()
        from transformers import T5ForConditionalGeneration as ProtoModel
        pretrained_model_name_or_path="uer/t5-base-chinese-cluecorpussmall"
    elif model_name == "mT5-base":
        from transformers import AutoModelForSeq2SeqLM as ProtoModel
        pretrained_model_name_or_path = "google/mt5-base"
    elif model_name == "Proto":
        if training_args.copy_weight == 0:
            print("Hint: Load Proto self-Distill Contrastive Bert (NAACL2022)")
            from models.bert.modeling_bert_v4 import ProtoModel_v3 as ProtoModel
        else:
            print("Hint: Load Proto COCO-LM (NIPS2022)")
            from models.bert.modeling_bert_v4 import ProtoModel_copy as ProtoModel
    elif model_name == "Gector":
        from models.bert.modeling_bert_v3 import GectorModel as ProtoModel
    elif model_name == "GPT":
        from transformers import GPT2LMHeadModel as ProtoModel
    elif model_name == "T5":
        from models.t5.modeling_t5 import T5ForConditionalGeneration as ProtoModel
        pretrained_model_name_or_path="uer/t5-base-chinese-cluecorpussmall"
    elif model_name == "mBART-50":
        print("Warning: mBART-50 is too large to train even in GTX3090[24G] (ctc task seq2seq 30w limit batch_size 16 cost 30+ hours)!")
        print("There is no base : https://github.com/facebookresearch/fairseq/issues/3252")
        exit()
        from transformers import MBartForConditionalGeneration as ProtoModel
        pretrained_model_name_or_path = "facebook/mbart-large-50"

    elif model_name is None or model_name == "BERT":
        if training_args.pretrained_name == "chinesebert":
            print("Hint: Load ChineseBert MaskedLM")
            from chinesebert import ChineseBertConfig, ChineseBertForMaskedLM 
            config = ChineseBertConfig.from_pretrained(pretrained_model_name_or_path)
            model = ChineseBertForMaskedLM.from_pretrained(pretrained_model_name_or_path, config=config)
            return model
        elif training_args.pretrained_name == "roformer":
            from roformer import RoFormerForMaskedLM
            model = RoFormerForMaskedLM.from_pretrained( pretrained_model_name_or_path )
            return model

        print("Hint: Load Default BertForMaskedLM.")
        from transformers import BertForMaskedLM as ProtoModel
    else:
        print(" Error: No such " + model_name)
        exit(0)

    if model_name != "Proto":
        model = ProtoModel.from_pretrained(pretrained_model_name_or_path=pretrained_model_name_or_path)
    else:
        #model = ProtoModel.from_pretrained(pretrained_model_name_or_path=pretrained_model_name_or_path)
        model = ProtoModel(
                        pretrained_model_name_or_path=pretrained_model_name_or_path, 
                        training_args=training_args,
                        )

    if not model:
        print(" Warning: wrong model name ! Check the core.py  ")
        exit()
    return model


def get_dataset(dataset, path_head):
    """
    Deprecation Warning !
    preprocess wrapped in load_ctc2021
    return : mydate
                torch.LongTensor
    
        Good day!
    """

    print("Loading Dataset !")
    exec("os.system('date')")

    if dataset == "ctc2021":
        train_data, eval_data, test_data = load_ctc2021()
    elif dataset == "sighan":
        train_data, eval_data, test_data = load_sighan(path_head=path_head)
    else:
        print("Error: No such dataset ")
        print(training_args.dataset)
        exit(0)

    train_dataset, eval_dataset, test_dataset = mydataset(train_data), mydataset(eval_data), mydataset(test_data)

    print("Loading Succeed !")
    exec("os.system('date')")

    return train_dataset, eval_dataset, test_dataset


def get_dataset_plus(training_args):
    """
    preprocess wrapped in load_ctc2021
    return : mydate
                torch.LongTensor
    
        Good day!
    """

    print("Loading Dataset !")
    ddp_exec("os.system('date')")

    if training_args.dataset == "ctc2021":
        train_data, eval_data, test_data = load_ctc2021(args=training_args)
    elif "sighan" in training_args.dataset:
        #train_data, eval_data, test_data = load_sighan(path_head)
        if training_args.model_name == "Gector":
            return _get_Gector_dataset(args=training_args)
        if training_args.pretrained_name == "chinesebert":
            if "mask" in training_args.dataset:
                return _get_chinesebert_mask_dataset(args=training_args)
            elif "holy" in training_args.dataset:
                return _get_chinesebert_holy_dataset(args=training_args)
            else:
                return _get_chinesebert_dataset(args=training_args)
        elif "holy" in training_args.dataset:
            if "mask" in training_args.dataset:
                return _get_holy_mask_dataset(args=training_args)
            else:
                return _get_holy_dataset(args=training_args)
        elif "mask" in training_args.dataset:
            return _get_mask_dataset(args=training_args)
        elif "holy" in training_args.dataset:
            return _get_holy_dataset(args=training_args)
        elif "enchanted" in training_args.dataset:
            return _get_enchanted_dataset(args=training_args)
        elif "raw" in training_args.dataset:
            return _get_raw_dataset(args=training_args)
        elif 'ReaLiSe' in training_args.dataset:
            return _get_ReaLiSe_dataset(args=training_args)
        elif 'expand' in training_args.dataset:
            return _get_expand_dataset(args=training_args)
        elif 'pure' in training_args.dataset:
            return _get_pure_dataset(args=training_args)
        else:
            print("Unclear data type, load default raw sighan")
            return _get_raw_dataset(args=training_args)
    else:
        print("Error: No such dataset ")
        print(training_args.dataset)
        exit(0)

    train_dataset, eval_dataset, test_dataset = mydataset(train_data), mydataset(eval_data), mydataset(test_data)

    print("Loading Succeed !")
    ddp_exec("os.system('date')")

    return train_dataset, eval_dataset, test_dataset


def _get_enchanted_dataset(args, which="15"):
    """
    Gector for sighan
    """
    print("Loading Enchanted Dataset !")
    ddp_exec("os.system('date')")

    train_data, eval_data, test_data = load_sighan_enchanted(path_head="")
    
    train_dataset, eval_dataset, test_dataset = mydataset(train_data), mydataset(eval_data), mydataset(test_data)

    print("Loaded successfully !")
    ddp_exec("os.system('date')")

    return train_dataset, eval_dataset, test_dataset


def _get_raw_dataset(args, which="15"):
    """
    Gector for sighan
    """
    print("Loading Raw Dataset !")
    ddp_exec("os.system('date')")

    train_data, eval_data, test_data = load_sighan(path_head="")
    
    train_dataset, eval_dataset, test_dataset = mydataset(train_data), mydataset(eval_data), mydataset(test_data)

    print("Loaded successfully !")
    ddp_exec("os.system('date')")

    return train_dataset, eval_dataset, test_dataset

def _get_pure_dataset(args, which="15"):
    """
    Gector for sighan
    """
    print("Loading Pure SIGHAN Dataset !")
    ddp_exec("os.system('date')")

    train_data, eval_data, test_data = load_sighan_pure(path_head="")
    
    train_dataset, eval_dataset, test_dataset = mydataset(train_data), mydataset(eval_data), mydataset(test_data)

    print("Loaded successfully !")
    ddp_exec("os.system('date')")

    return train_dataset, eval_dataset, test_dataset

def _get_holy_dataset(args, which="15"):
    """
    Holy for sighan
    """
    print("Loading Holy Dataset !")
    ddp_exec("os.system('date')")

    train_data, eval_data, test_data = load_sighan_holy(path_head="")
    
    train_dataset, eval_dataset, test_dataset = mydataset(train_data), mydataset(eval_data), mydataset(test_data)

    print("Loaded successfully !")
    ddp_exec("os.system('date')")

    return train_dataset, eval_dataset, test_dataset

def _get_holy_mask_dataset(args, which="15"):
    """
    Holy Mask for sighan
    """
    print("Loading Holy Mask Dataset !")
    ddp_exec("os.system('date')")

    train_data, eval_data, test_data = load_sighan_holy_mask(path_head="")
    
    train_dataset, eval_dataset, test_dataset = mydataset(train_data), mydataset(eval_data), mydataset(test_data)

    print("Loaded successfully !")
    ddp_exec("os.system('date')")

    return train_dataset, eval_dataset, test_dataset


def _get_mask_dataset(args, which="15"):
    """
    Gector for sighan
    """
    print("Loading MASK Dataset !")
    ddp_exec("os.system('date')")

    train_data, eval_data, test_data = load_sighan_mask(path_head="")
    
    train_dataset, eval_dataset, test_dataset = mydataset(train_data), mydataset(eval_data), mydataset(test_data)

    print("Loaded successfully !")
    ddp_exec("os.system('date')")

    return train_dataset, eval_dataset, test_dataset


def _get_Gector_dataset(args, which="15"):
    """
    Gector for sighan
    """
    print("Loading GECTOR Dataset !")
    ddp_exec("os.system('date')")

    train_data, eval_data, test_data = load_sighan_gector(path_head="")
    
    train_dataset, eval_dataset, test_dataset = mydataset(train_data), mydataset(eval_data), mydataset(test_data)

    print("Loaded successfully !")
    ddp_exec("os.system('date')")

    return train_dataset, eval_dataset, test_dataset

def _get_chinesebert_holy_dataset(args, which="15"):
    """
    ChineseBert for sighan
    Mainly diff in no max_length and 'pinyin_idx' must be 8 * len('input_ids')
    """
    print("Loading ChineseBert Holy Dataset !")
    ddp_exec("os.system('date')")

    train_data, eval_data, test_data = load_sighan_chinesebert_holy(path_head="")
    
    train_dataset, eval_dataset, test_dataset = mydataset(train_data), mydataset(eval_data), mydataset(test_data)

    print("Loaded successfully !")
    ddp_exec("os.system('date')")

    return train_dataset, eval_dataset, test_dataset

def _get_chinesebert_dataset(args, which="15"):
    """
    ChineseBert for sighan
    Mainly diff in no max_length and 'pinyin_idx' must be 8 * len('input_ids')
    """
    print("Loading ChineseBert Dataset !")
    ddp_exec("os.system('date')")

    train_data, eval_data, test_data = load_sighan_chinesebert(path_head="")
    
    train_dataset, eval_dataset, test_dataset = mydataset(train_data), mydataset(eval_data), mydataset(test_data)

    print("Loaded successfully !")
    ddp_exec("os.system('date')")

    return train_dataset, eval_dataset, test_dataset

def _get_chinesebert_mask_dataset(args, which="15"):
    """
    ChineseBert for sighan
    Mainly diff in no max_length and 'pinyin_idx' must be 8 * len('input_ids')
    """
    print("Loading Masked ChineseBert Dataset !")
    ddp_exec("os.system('date')")

    train_data, eval_data, test_data = load_sighan_chinesebert_mask(path_head="")
    
    train_dataset, eval_dataset, test_dataset = mydataset(train_data), mydataset(eval_data), mydataset(test_data)

    print("Loaded successfully !")
    ddp_exec("os.system('date')")

    return train_dataset, eval_dataset, test_dataset


def _get_ReaLiSe_dataset(args, which="15"):
    """
    For its 
    """
    print("Loading ReaLiSe Dataset !")
    print("Hint: The Data You loading now is the preprocessed sighan from ReaLise, ")
    ddp_exec("os.system('date')")

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

    print("Loaded successfully !")
    ddp_exec("os.system('date')")
    print("over")
    return trans2mydataset(train_dataset), trans2mydataset(eval_dataset), trans2mydataset(test_dataset)

def _get_expand_dataset(args, which="15"):
    """
    NLPcc and HSK Expand for sighan
    """
    print("Loading Expand Dataset !")
    ddp_exec("os.system('date')")

    train_data, eval_data, test_data = load_sighan_expand(path_head="")
    
    train_dataset, eval_dataset, test_dataset = mydataset(train_data), mydataset(eval_data), mydataset(test_data)

    print("Loaded successfully !")
    ddp_exec("os.system('date')")

    return train_dataset, eval_dataset, test_dataset

def _get_sighan_test(args, which, path_head=""):
    """
    """
    print("Loading Dataset !")
    ddp_exec("os.system('date')")
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
    ddp_exec("os.system('date')")

    return test_dataset


def _get_lattice_dataset(args, dataset="sighan", path_head="."):
    """
    """
    print("Loading Dataset !")
    ddp_exec("os.system('date')")

    if dataset == "sighan":
        datasets, vocabs, embeddings = load_lattice_sighan(path_head=path_head)
    else:
        exit()

    datasets["train"], datasets["valid"], datasets["test"] = mydataset(datasets["train"]), mydataset(datasets["valid"]), mydataset(datasets["test"])

    return datasets, vocabs, embeddings


def _get_magic_plus_dataset(args, dataset="sighan", path_head=""):
    """
    """
    print("Loading Dataset !")
    ddp_exec("os.system('date')")

    if dataset == "sighan":
        train_data, eval_data, test_data = load_abs_pos_sighan_plus(path_head=path_head)
    else:
        exit()

    train_dataset, eval_dataset, test_dataset = mydataset(train_data), mydataset(eval_data), mydataset(test_data)

    print("Loading Succeed !")

    ddp_exec("os.system('date')")

    return train_dataset, eval_dataset, test_dataset


def _get_magic_dataset(args, dataset="sighan", path_head=""):
    """
    """
    print("Loading Dataset !")
    ddp_exec("os.system('date')")

    if dataset == "sighan":
        train_data, eval_data, test_data = load_abs_pos_sighan(path_head=path_head)
    else:
        exit()

    train_dataset, eval_dataset, test_dataset = mydataset(train_data), mydataset(eval_data), mydataset(test_data)

    print("Loading Succeed !")

    ddp_exec("os.system('date')")

    return train_dataset, eval_dataset, test_dataset


def _get_magic_lang8_dataset(args, dataset="sighan", path_head=""):
    """
    """
    print("Loading Dataset !")
    ddp_exec("os.system('date')")

    if dataset == "sighan":
        train_data, eval_data, test_data = load_abs_pos_sighan_lang8(path_head=path_head)
    else:
        exit()

    train_dataset, eval_dataset, test_dataset = mydataset(train_data), mydataset(eval_data), mydataset(test_data)

    print("Loading Succeed !")

    ddp_exec("os.system('date')")

    return train_dataset, eval_dataset, test_dataset


def _get_magic_expand_dataset(args, dataset="sighan", path_head=""):
    """
    """
    print("Loading Dataset !")
    ddp_exec("os.system('date')")

    if dataset == "sighan":
        train_data, eval_data, test_data = load_abs_pos_sighan_plus(path_head=path_head)
    else:
        exit()

    train_dataset, eval_dataset, test_dataset = mydataset(train_data), mydataset(eval_data), mydataset(test_data)

    print("Loading Succeed !")

    ddp_exec("os.system('date')")

    return train_dataset, eval_dataset, test_dataset


def _get_super_magic_dataset(args, dataset="sighan", path_head=""):
    """
    """
    print("Loading Dataset !")
    ddp_exec("os.system('date')")

    if dataset == "sighan":
        train_data, eval_data, test_data, tokenizer = load_abs_pos_and_spe_token_sighan(path_head=path_head)
    else:
        exit()

    train_dataset, eval_dataset, test_dataset = mydataset(train_data), mydataset(eval_data), mydataset(test_data)

    print("Loading Succeed !")

    ddp_exec("os.system('date')")

    return train_dataset, eval_dataset, test_dataset, tokenizer


def get_metrics(training_args):
    if "sighan" in training_args.dataset:
        print("Hint: Using aligned sighan F1_score as metric")
        return _get_metrics(training_args)
    if "ctc2021" in training_args.dataset :
        print("Hint: Using Seq2Seq ctc2021 score as metric")
        return _get_seq2seq_metrics(training_args)
    else:
        print("Error when getting metrics.")
        exit(0)

class mini_reporter:
    def __init__(self):
        self.messages = []
    
    def record(self, message):
        self.messages.append(message)

    def out(self, obj):
        from collections import Iterable
        if isinstance(obj, Iterable):
            for e in obj:
                print(e)
        else:
            print(obj)
        
    def report(self):
        print("[Reporter] ")
        for message in self.messages:
            self.out(message)
        print("[Over]")

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

        reporter = mini_reporter()

        sources, preds, labels = eval_preds# (num, length) np.array
 
        tp, fp, fn = 0, 0, 0

        sent_p, sent_n = 0, 0

        for i in range(len(sources)):
            #print(sources[i])
            #print(preds[i])
            #print(labels[i])

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
            #print(source)
            #print(pred)
            #print(label)
            #print((pred != source).any())
            #print((pred == label).all())
            #print((label != source).any())

            reporter.record([pred, source, label])
            reporter.record([(pred!=source).any(), (pred==label).all()])

            if training_args.model_name != "Gector":
                # label: [101, 2,... 3, 102]
                if (pred != source).any():
                    sent_p += 1
                    #print("sent_p")
                    if (pred == label).all():
                        tp += 1
                        # print("tp")

                if (label != source).any():
                    sent_n += 1
                    #print("sent_n")
            else:
                # label : [ 1,1,1,1,1 ]
                if (pred != 1).any():
                    sent_p += 1

                    if (pred == label).all():
                        tp += 1
            
                if (label != 1).any():
                    sent_n += 1

        #print(tp, sent_p, sent_n)

        precision = tp / (sent_p + 1e-10)

        recall = tp / (sent_n + 1e-10)

        F1_score = 2 * precision * recall / (precision + recall + 1e-10)

        Turtle = time.time() - Achilles

        if F1_score < 0.05:
            print("Warning : metric score is too Low (< 0.05), maybe something goes wrong, check your codes please.")
            #reporter.report()
            #exit(0)
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

        return {"eval_F1_score":float( 0.8 * F1_score_detect + 0.2 * F1_score_correct ), 
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

