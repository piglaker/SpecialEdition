
from asyncore import read
import re
import json
from typing import Dict, List
import itertools

from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    MBart50TokenizerFast,
)
import jieba

import sys
sys.path.append("/remote-home/xtzhang/CTC/CTC2021/SpecialEdition")


from utils.io import read_csv


def wash_n(data):
    import re
    return [re.sub("\n", "", i) for i in data ]

def load_json(data_path):
    
    file = open(data_path, 'r', encoding='utf-8')

    padded_data = []

    for line in file.readlines():
        tmp = json.loads(line)
        tmp_dict = {}
        tmp_dict["original_text"] = tmp["source"]
        tmp_dict["correct_text"] = tmp["target"]
        padded_data.append(tmp_dict)

    return padded_data

def load_tmp():
    train_path = "./data/tmp/train/train.csv"
    test_path = "./data/tmp/test/test.csv"

    train_source = read_csv(train_path)
    test_source = read_csv(test_path)

    def preprocess(data):
        src, tgt = [], []

        for i in data[1:]:
            _id, _label, _source =  i.split()
            src.append(_source)
            tgt.append(_label)    

        return src, tgt
    
    train_source, train_target = preprocess(train_source[1000:])

    valid_source, valid_target = preprocess(train_source[:1000])

    test_source, test_target = preprocess(test_source)

    tokenizer_model_name_path="hfl/chinese-roberta-wwm-ext"

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_model_name_path)

    ####
    # some shit ->
    ####
    
    train_source_tok = tokenizer.batch_encode_plus(train_source, return_token_type_ids=False)#seems transformers max_length not work
    #train_target_tok = tokenizer.batch_encode_plus(train_target, return_token_type_ids=False)#remove padding=True, max_length=512
    valid_source_tok = tokenizer.batch_encode_plus(valid_source, return_token_type_ids=False)
    #valid_target_tok = tokenizer.batch_encode_plus(valid_target, return_token_type_ids=False)
    test_source_tok = tokenizer.batch_encode_plus(test_source, return_token_type_ids=False)
    #test_target_tok = tokenizer.batch_encode_plus(test_target, return_token_type_ids=False)

    train_source_tok["labels"] = train_target#train_target_tok["input_ids"]
    valid_source_tok["labels"] = valid_target#valid_target_tok["input_ids"]
    test_source_tok["labels"] = test_target#test_target_tok["input_ids"]

    def transpose(inputs):
        features = []
        for i in tqdm(range(len(inputs["input_ids"]))):
            features.append({key:inputs[key][i][:128] for key in inputs.keys()}) #we fix here (truncation 
        return features 

    return transpose(train_source_tok), transpose(valid_source_tok), transpose(test_source_tok)

from fastNLP import cache_results
@cache_results(_cache_fp='cache/ctc2021_extra', _refresh=False)
def load_ctc2021(args):

    print("[Data] Loading CTC2021 Dataset")

    #print("[Data] #"*5+ " Loading toy datasets for debugging ... " + '#'*5)
    train_source_path = "./data/rawdata/ctc2021/train.src"
    train_target_path = "./data/rawdata/ctc2021/train.tgt"
    valid_source_path = "./data/rawdata/ctc2021/valid.src"
    valid_target_path = "./data/rawdata/ctc2021/valid.tgt"
    test_source_path = "./data/rawdata/ctc2021/test.src"
    test_target_path = "./data/rawdata/ctc2021/test.tgt"

    train_source = read_csv(train_source_path)    
    train_target = read_csv(train_target_path)
    valid_source = read_csv(valid_source_path)
    valid_target = read_csv(valid_target_path)
    test_source = read_csv(test_source_path)
    test_target = read_csv(test_target_path)

    if args.use_extra_dataset:
        print("[Data] Using Large v2 & pseudo as extra train set")

        train_v2_source_path = "./data/rawdata/ctc2021/train_v2.src"
        train_v2_target_path = "./data/rawdata/ctc2021/train_v2.tgt"
        
        train_source += read_csv(train_v2_source_path)    
        train_target += read_csv(train_v2_target_path)
        
        # train_v3_source_path = "./data/rawdata/ctc2021/train_v3.src"
        # train_v3_target_path = "./data/rawdata/ctc2021/train_v3.tgt"

        # train_source += read_csv(train_v3_source_path)    
        # train_target += read_csv(train_v3_target_path) 

    tokenizer_model_name_path="hfl/chinese-roberta-wwm-ext"

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_model_name_path)

    #if args.model_name == "mBART-50":
    #    tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50", src_lang="zh_CN", tgt_lang="zh_CN")
    #elif args.model_name == "mT5-base":
    #    tokenizer = AutoTokenizer.from_pretrained("google/mt5-base")

    ####
    # some shit ->
    ####
    
    train_source_tok = tokenizer.batch_encode_plus(train_source, return_token_type_ids=False)#seems transformers max_length not work
    train_target_tok = tokenizer.batch_encode_plus(train_target, return_token_type_ids=False)#remove padding=True, max_length=512
    valid_source_tok = tokenizer.batch_encode_plus(valid_source, return_token_type_ids=False)
    valid_target_tok = tokenizer.batch_encode_plus(valid_target, return_token_type_ids=False)
    test_source_tok = tokenizer.batch_encode_plus(test_source, return_token_type_ids=False)
    test_target_tok = tokenizer.batch_encode_plus(test_target, return_token_type_ids=False)

    train_source_tok["labels"] = train_target_tok["input_ids"]
    valid_source_tok["labels"] = valid_target_tok["input_ids"]
    test_source_tok["labels"] = test_target_tok["input_ids"]

    def transpose(inputs):
        features = []
        for i in tqdm(range(len(inputs["input_ids"]))):
            features.append({key:inputs[key][i][:128] for key in inputs.keys()}) #we fix here (truncation 
        return features 

    return transpose(train_source_tok), transpose(valid_source_tok), transpose(test_source_tok)


from fastNLP import cache_results
@cache_results(_cache_fp='cache/ctc2021_mT5_extra', _refresh=False)
def load_ctc2021_mT5(args):

    print("[Data] Loading CTC2021 for mT5 Dataset")

    #print("[Data] #"*5+ " Loading toy datasets for debugging ... " + '#'*5)
    train_source_path = "./data/rawdata/ctc2021/train.src"
    train_target_path = "./data/rawdata/ctc2021/train.tgt"
    valid_source_path = "./data/rawdata/ctc2021/valid.src"
    valid_target_path = "./data/rawdata/ctc2021/valid.tgt"
    test_source_path = "./data/rawdata/ctc2021/test.src"
    test_target_path = "./data/rawdata/ctc2021/test.tgt"

    train_source = read_csv(train_source_path)    
    train_target = read_csv(train_target_path)
    valid_source = read_csv(valid_source_path)
    valid_target = read_csv(valid_target_path)
    test_source = read_csv(test_source_path)
    test_target = read_csv(test_target_path)

    if args.use_extra_dataset:
        print("[Data] Using Large v2 & pseudo as extra train set")

        train_v2_source_path = "./data/rawdata/ctc2021/train_v2.src"
        train_v2_target_path = "./data/rawdata/ctc2021/train_v2.tgt"
        
        train_source += read_csv(train_v2_source_path)    
        train_target += read_csv(train_v2_target_path)
        
        # train_v3_source_path = "./data/rawdata/ctc2021/train_v3.src"
        # train_v3_target_path = "./data/rawdata/ctc2021/train_v3.tgt"

        # train_source += read_csv(train_v3_source_path)    
        # train_target += read_csv(train_v3_target_path) 

    tokenizer_model_name_path="hfl/chinese-roberta-wwm-ext"

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_model_name_path)

    tokenizer = AutoTokenizer.from_pretrained( "google/" + args.model_name )

    #train_source = tokenizer(train_source, text_target=train_target, return_tensors="pt")

    ####
    # some shit ->
    ####
    
    train_source_tok = tokenizer.batch_encode_plus(train_source, return_token_type_ids=False)#seems transformers max_length not work
    train_target_tok = tokenizer.batch_encode_plus(train_target, return_token_type_ids=False)#remove padding=True, max_length=512
    valid_source_tok = tokenizer.batch_encode_plus(valid_source, return_token_type_ids=False)
    valid_target_tok = tokenizer.batch_encode_plus(valid_target, return_token_type_ids=False)
    test_source_tok = tokenizer.batch_encode_plus(test_source, return_token_type_ids=False)
    test_target_tok = tokenizer.batch_encode_plus(test_target, return_token_type_ids=False)

    train_source_tok["labels"] = train_target_tok["input_ids"]
    valid_source_tok["labels"] = valid_target_tok["input_ids"]
    test_source_tok["labels"] = test_target_tok["input_ids"]

    def transpose(inputs):
        features = []
        for i in tqdm(range(len(inputs["input_ids"]))):
            features.append({key:inputs[key][i][:128] for key in inputs.keys()}) #we fix here (truncation 
        return features 

    return transpose(train_source_tok), transpose(valid_source_tok), transpose(test_source_tok)

from fastNLP import cache_results
@cache_results(_cache_fp='cache/sighan_raw', _refresh=False)
def load_sighan(path_head=""):

    print("[Data] Loading SigHan Dataset ...")

    train_source_path = path_head + "./data/rawdata/sighan/raw/train.src"
    train_target_path = path_head + "./data/rawdata/sighan/raw/train.tgt"
    valid_source_path = path_head + "./data/rawdata/sighan/raw/valid.src"
    valid_target_path = path_head + "./data/rawdata/sighan/raw/valid.tgt"#valid should be same to test ( sighan 15
    test_source_path = path_head + "./data/rawdata/sighan/raw/test.src"
    test_target_path = path_head + "./data/rawdata/sighan/raw/test.tgt"

    train_source = read_csv(train_source_path)#[:2000]#[274144:]#for only sighan
    train_target = read_csv(train_target_path)#[:2000]#[274144:]
    
    valid_source = read_csv(valid_source_path)
    valid_target = read_csv(valid_target_path)

    test_source = read_csv(test_source_path)
    test_target = read_csv(test_target_path)

    #valid_source = train_source[:1100]#for valid overfit problem
    #valid_target = train_target[:1100]
    #train_source = train_source[1100:]
    #train_target = train_target[1100:]

    tokenizer_model_name_path="hfl/chinese-roberta-wwm-ext"

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_model_name_path)

    train_source_tok = tokenizer.batch_encode_plus(train_source, return_token_type_ids=False)#seems transformers max_length not work
    train_target_tok = tokenizer.batch_encode_plus(train_target, return_token_type_ids=False)#remove padding=True, max_length=512
    valid_source_tok = tokenizer.batch_encode_plus(valid_source, return_token_type_ids=False)
    valid_target_tok = tokenizer.batch_encode_plus(valid_target, return_token_type_ids=False)
    test_source_tok = tokenizer.batch_encode_plus(test_source, return_token_type_ids=False)
    test_target_tok = tokenizer.batch_encode_plus(test_target, return_token_type_ids=False)

    train_source_tok["labels"] = train_target_tok["input_ids"]
    valid_source_tok["labels"] = valid_target_tok["input_ids"]
    test_source_tok["labels"] = test_target_tok["input_ids"]

    def transpose(inputs):
        features = []
        for i in tqdm(range(len(inputs["input_ids"]))):
            #ugly fix for encoder model (the same length
            features.append({key:inputs[key][i][:128] for key in inputs.keys()}) #we fix here (truncation 
        return features 

    return transpose(train_source_tok), transpose(valid_source_tok), transpose(test_source_tok)

from fastNLP import cache_results
@cache_results(_cache_fp='cache/sighan_pure', _refresh=True)
def load_sighan_pure(path_head=""):

    print("[Data] Loading SigHan Pure Dataset ...")

    train_source_path = path_head + "./data/rawdata/sighan/raw/train.src"
    train_target_path = path_head + "./data/rawdata/sighan/raw/train.tgt"
    valid_source_path = path_head + "./data/rawdata/sighan/raw/valid.src"
    valid_target_path = path_head + "./data/rawdata/sighan/raw/valid.tgt"#valid should be same to test ( sighan 15
    test_source_path = path_head + "./data/rawdata/sighan/raw/test.src"
    test_target_path = path_head + "./data/rawdata/sighan/raw/test.tgt"

    train_source = read_csv(train_source_path)[-51000:]#[:2000]#[274144:]#for only sighan
    train_target = read_csv(train_target_path)[-51000:]#[:2000]#[274144:]
    
    valid_source = read_csv(valid_source_path)
    valid_target = read_csv(valid_target_path)

    test_source = read_csv(test_source_path)
    test_target = read_csv(test_target_path)

    #valid_source = train_source[:1100]#for valid overfit problem
    #valid_target = train_target[:1100]
    #train_source = train_source[1100:]
    #train_target = train_target[1100:]

    tokenizer_model_name_path="hfl/chinese-roberta-wwm-ext"

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_model_name_path)

    train_source_tok = tokenizer.batch_encode_plus(train_source, return_token_type_ids=False)#seems transformers max_length not work
    train_target_tok = tokenizer.batch_encode_plus(train_target, return_token_type_ids=False)#remove padding=True, max_length=512
    valid_source_tok = tokenizer.batch_encode_plus(valid_source, return_token_type_ids=False)
    valid_target_tok = tokenizer.batch_encode_plus(valid_target, return_token_type_ids=False)
    test_source_tok = tokenizer.batch_encode_plus(test_source, return_token_type_ids=False)
    test_target_tok = tokenizer.batch_encode_plus(test_target, return_token_type_ids=False)

    train_source_tok["labels"] = train_target_tok["input_ids"]
    valid_source_tok["labels"] = valid_target_tok["input_ids"]
    test_source_tok["labels"] = test_target_tok["input_ids"]

    def transpose(inputs):
        features = []
        for i in tqdm(range(len(inputs["input_ids"]))):
            #ugly fix for encoder model (the same length
            features.append({key:inputs[key][i][:128] for key in inputs.keys()}) #we fix here (truncation 
        return features 

    return transpose(train_source_tok), transpose(valid_source_tok), transpose(test_source_tok)

from fastNLP import cache_results
@cache_results(_cache_fp='cache/sighan_expand', _refresh=False)
def load_sighan_expand(path_head=""):
    """
    """
    print("[Data] NLPCC_and HSK only for UNALIGNED Correction!")

    exit()
    
    print("[Data] Loading Expand SigHan Dataset ...")

    train_source_path = path_head + "./data/rawdata/sighan/raw/train.src"
    train_target_path = path_head + "./data/rawdata/sighan/raw/train.tgt"
    valid_source_path = path_head + "./data/rawdata/sighan/raw/valid.src"
    valid_target_path = path_head + "./data/rawdata/sighan/raw/valid.tgt"#valid should be same to test ( sighan 15
    test_source_path = path_head + "./data/rawdata/sighan/raw/test.src"
    test_target_path = path_head + "./data/rawdata/sighan/raw/test.tgt"

    expand_source_path = path_head + "./data/nlpcc_and_hsk/train.src"
    expand_target_path = path_head + "./data/nlpcc_and_hsk/train.trg"

    train_source = read_csv(train_source_path)#[:2000]#[274144:]#for only sighan
    train_target = read_csv(train_target_path)#[:2000]#[274144:]
    
    valid_source = read_csv(valid_source_path)
    valid_target = read_csv(valid_target_path)

    test_source = read_csv(test_source_path)
    test_target = read_csv(test_target_path)

    expand_source = read_csv(expand_source_path, remove_blank=True)
    expand_target = read_csv(expand_target_path, remove_blank=True)

    tokenizer_model_name_path="hfl/chinese-roberta-wwm-ext"

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_model_name_path)

    train_source_tok = tokenizer.batch_encode_plus(train_source + expand_source, return_token_type_ids=False)#seems transformers max_length not work
    train_target_tok = tokenizer.batch_encode_plus(train_target + expand_target, return_token_type_ids=False)#remove padding=True, max_length=512
    valid_source_tok = tokenizer.batch_encode_plus(valid_source, return_token_type_ids=False)
    valid_target_tok = tokenizer.batch_encode_plus(valid_target, return_token_type_ids=False)
    test_source_tok = tokenizer.batch_encode_plus(test_source, return_token_type_ids=False)
    test_target_tok = tokenizer.batch_encode_plus(test_target, return_token_type_ids=False)

    train_source_tok["labels"] = train_target_tok["input_ids"]
    valid_source_tok["labels"] = valid_target_tok["input_ids"]
    test_source_tok["labels"] = test_target_tok["input_ids"]

    def transpose(inputs):
        features = []
        for i in tqdm(range(len(inputs["input_ids"]))):
            #ugly fix for encoder model (the same length
            features.append({key:inputs[key][i][:128] for key in inputs.keys()}) #we fix here (truncation 
        return features 

    return transpose(train_source_tok), transpose(valid_source_tok), transpose(test_source_tok)

from fastNLP import cache_results
@cache_results(_cache_fp='cache/sighan_gector', _refresh=False)
def load_sighan_gector(path_head=""):
    """
    Tokenizer : 
        batch_encode_plus cause a problem that "笑嘻嘻中了set" would be tokenized to [1,2,3,4,5,6] ("set"'s token is 6 ),
        so the length of source.tok dismatch the label ( seq2tag generate bu scripts/sighan_v1/generate_vocab.py)
        I use ugly double for to solve this .
    """
    print("[Data] Loading SigHan GECTOR Dataset ...")

    train_source_path = path_head + "./data/rawdata/sighan/gector/train.src"
    train_target_path = path_head + "./data/rawdata/sighan/gector/train.tgt"
    valid_source_path = path_head + "./data/rawdata/sighan/gector/valid.src"
    valid_target_path = path_head + "./data/rawdata/sighan/gector/valid.tgt"#valid should be same to test ( sighan 15
    test_source_path = path_head + "./data/rawdata/sighan/gector/test.src"
    test_target_path = path_head + "./data/rawdata/sighan/gector/test.tgt"

    train_source = wash_n(read_csv(train_source_path))#[:2000]#[274144:]#for only sighan
    train_target = wash_n(read_csv(train_target_path))#[:2000]#[274144:]
    
    valid_source = wash_n(read_csv(valid_source_path))
    valid_target = wash_n(read_csv(valid_target_path))

    test_source = wash_n(read_csv(test_source_path))
    test_target = wash_n(read_csv(test_target_path))

    tokenizer_model_name_path="hfl/chinese-roberta-wwm-ext"

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_model_name_path)

    def my_batch_encode_plus(source):
        """
        Almost squally to batch_encode_plus
        """
        new_source = { "input_ids":[], "attention_mask":[], "labels":[] }

        for sentence in source:
            tmp = [101]
            for i in sentence:
                tmp.append(tokenizer.convert_tokens_to_ids(i))
            
            tmp.append(102)
            new_source["input_ids"].append(tmp)
            new_source["attention_mask"].append( [ 1 for i in tmp] )

        return new_source

    train_source_tok = my_batch_encode_plus(train_source)#seems transformers max_length not work
    #train_target_tok = tokenizer.batch_encode_plus(train_target, return_token_type_ids=False)#remove padding=True, max_length=512
    valid_source_tok = my_batch_encode_plus(valid_source)
    #valid_target_tok = tokenizer.batch_encode_plus(valid_target, return_token_type_ids=False)
    test_source_tok = my_batch_encode_plus(test_source)
    #test_target_tok = tokenizer.batch_encode_plus(test_target, return_token_type_ids=False)

    def preprocess(s):
        return list(map(int, s.split(",")))

    train_source_tok["labels"] = list(map(preprocess, train_target))
    valid_source_tok["labels"] = list(map(preprocess, valid_target))
    test_source_tok["labels"] = list(map(preprocess, test_target))

    def transpose(inputs):
        features = []
        for i in tqdm(range(len(inputs["input_ids"]))):
            #ugly fix for encoder model (the same length
            features.append({key:inputs[key][i][:128] for key in inputs.keys()}) #we fix here (truncation 
        return features 
    return transpose(train_source_tok), transpose(valid_source_tok), transpose(test_source_tok)

from fastNLP import cache_results
@cache_results(_cache_fp='cache/sighan_chinesebert', _refresh=False)
def load_sighan_chinesebert(path_head=""):
    """
    Tokenizer : 
        import torch
        from chinesebert import ChineseBertForMaskedLM, ChineseBertTokenizerFast, ChineseBertConfig
        pretrained_model_name = "junnyu/ChineseBERT-base"

        tokenizer = ChineseBertTokenizerFast.from_pretrained(pretrained_model_name)
        chinese_bert = ChineseBertForMaskedLM.from_pretrained(pretrained_model_name)

        text = "北京是[MASK]国的首都。"
        inputs = tokenizer(text, return_tensors="pt")

        with torch.no_grad():
            o = chinese_bert(**inputs)
            l = o.logits
    """
    print("[Data] Loading SigHan ChineseBert Dataset ...")

    train_source_path = path_head + "./data/rawdata/sighan/raw/train.src"
    train_target_path = path_head + "./data/rawdata/sighan/raw/train.tgt"
    valid_source_path = path_head + "./data/rawdata/sighan/raw/valid.src"
    valid_target_path = path_head + "./data/rawdata/sighan/raw/valid.tgt"#valid should be same to test ( sighan 15
    test_source_path = path_head + "./data/rawdata/sighan/raw/test.src"
    test_target_path = path_head + "./data/rawdata/sighan/raw/test.tgt"

    train_source = read_csv(train_source_path)#[:2000]#[274144:]#for only sighan
    train_target = read_csv(train_target_path)#[:2000]#[274144:]
    
    valid_source = read_csv(valid_source_path)
    valid_target = read_csv(valid_target_path)

    test_source = read_csv(test_source_path)
    test_target = read_csv(test_target_path)

    from chinesebert import ChineseBertTokenizerFast
    pretrained_model_name = "junnyu/ChineseBERT-base"

    tokenizer = ChineseBertTokenizerFast.from_pretrained(pretrained_model_name)

    train_source_tok = tokenizer(train_source, padding=True, truncation=True, max_length=128)
    train_target_tok = tokenizer(train_target, padding=True, truncation=True, max_length=128) 
    valid_source_tok = tokenizer(valid_source, padding=True, truncation=True, max_length=128)
    valid_target_tok = tokenizer(valid_target, padding=True, truncation=True, max_length=128) 
    test_source_tok = tokenizer(test_source, padding=True, truncation=True, max_length=128) 
    test_target_tok = tokenizer(test_target, padding=True, truncation=True, max_length=128) 

    train_source_tok["labels"] = train_target_tok["input_ids"]
    valid_source_tok["labels"] = valid_target_tok["input_ids"]
    test_source_tok["labels"] = test_target_tok["input_ids"]

    def transpose(inputs):
        features = []
        for i in tqdm(range(len(inputs["input_ids"]))):
            #ugly fix for encoder model (the same length
            features.append({key:inputs[key][i] for key in inputs.keys()}) 
        return features 

    return transpose(train_source_tok), transpose(valid_source_tok), transpose(test_source_tok)

from fastNLP import cache_results
@cache_results(_cache_fp='cache/sighan_chinesebert_holy', _refresh=False)
def load_sighan_chinesebert_holy(path_head=""):
    """
        Holy
    """
    print("[Data] Loading Holy SigHan ChineseBert Dataset ...")

    train_source_path = path_head + "./data/rawdata/sighan/holy/train.src"
    train_target_path = path_head + "./data/rawdata/sighan/holy/train.tgt"
    valid_source_path = path_head + "./data/rawdata/sighan/holy/valid.src"
    valid_target_path = path_head + "./data/rawdata/sighan/holy/valid.tgt"#valid should be same to test ( sighan 15
    test_source_path = path_head + "./data/rawdata/sighan/holy/test.src"
    test_target_path = path_head + "./data/rawdata/sighan/holy/test.tgt"

    train_source = read_csv(train_source_path)#[:2000]#[274144:]#for only sighan
    train_target = read_csv(train_target_path)#[:2000]#[274144:]
    
    valid_source = read_csv(valid_source_path)
    valid_target = read_csv(valid_target_path)

    test_source = read_csv(test_source_path)
    test_target = read_csv(test_target_path)

    from chinesebert import ChineseBertTokenizerFast
    pretrained_model_name = "junnyu/ChineseBERT-base"

    tokenizer = ChineseBertTokenizerFast.from_pretrained(pretrained_model_name)

    train_source_tok = tokenizer(train_source, padding=True, truncation=True, max_length=128)
    train_target_tok = tokenizer(train_target, padding=True, truncation=True, max_length=128) 
    valid_source_tok = tokenizer(valid_source, padding=True, truncation=True, max_length=128)
    valid_target_tok = tokenizer(valid_target, padding=True, truncation=True, max_length=128) 
    test_source_tok = tokenizer(test_source, padding=True, truncation=True, max_length=128) 
    test_target_tok = tokenizer(test_target, padding=True, truncation=True, max_length=128) 

    train_source_tok["labels"] = train_target_tok["input_ids"]
    valid_source_tok["labels"] = valid_target_tok["input_ids"]
    test_source_tok["labels"] = test_target_tok["input_ids"]

    def transpose(inputs):
        features = []
        for i in tqdm(range(len(inputs["input_ids"]))):
            #ugly fix for encoder model (the same length
            features.append({key:inputs[key][i] for key in inputs.keys()}) #we fix here (truncation 
        return features 

    return transpose(train_source_tok), transpose(valid_source_tok), transpose(test_source_tok)

from fastNLP import cache_results
@cache_results(_cache_fp='cache/sighan_chinesebert_mask', _refresh=False)
def load_sighan_chinesebert_mask(path_head=""):
    """
    """
    print("[Data] Loading SigHan ChineseBert Dataset ...")

    train_source_path = path_head + "./data/rawdata/sighan/raw/train.src"
    train_target_path = path_head + "./data/rawdata/sighan/raw/train.tgt"
    valid_source_path = path_head + "./data/rawdata/sighan/raw/valid.src"
    valid_target_path = path_head + "./data/rawdata/sighan/raw/valid.tgt"#valid should be same to test ( sighan 15
    test_source_path = path_head + "./data/rawdata/sighan/raw/test.src"
    test_target_path = path_head + "./data/rawdata/sighan/raw/test.tgt"

    train_source = read_csv(train_source_path)#[:2000]#[274144:]#for only sighan
    train_target = read_csv(train_target_path)#[:2000]#[274144:]
    
    valid_source = read_csv(valid_source_path)
    valid_target = read_csv(valid_target_path)

    test_source = read_csv(test_source_path)
    test_target = read_csv(test_target_path)

    from chinesebert import ChineseBertTokenizerFast
    pretrained_model_name = "junnyu/ChineseBERT-base"

    tokenizer = ChineseBertTokenizerFast.from_pretrained(pretrained_model_name)

    train_source_tok = tokenizer(train_source, padding=True, truncation=True, max_length=128)
    train_target_tok = tokenizer(train_target, padding=True, truncation=True, max_length=128) 
    valid_source_tok = tokenizer(valid_source, padding=True, truncation=True, max_length=128)
    valid_target_tok = tokenizer(valid_target, padding=True, truncation=True, max_length=128) 
    test_source_tok = tokenizer(test_source, padding=True, truncation=True, max_length=128) 
    test_target_tok = tokenizer(test_target, padding=True, truncation=True, max_length=128) 

    train_source_tok["labels"] = train_target_tok["input_ids"]
    valid_source_tok["labels"] = valid_target_tok["input_ids"]
    test_source_tok["labels"] = test_target_tok["input_ids"]

    # a ugly mask to replace the 
    def mask(source):
        for i in range(len(source)):
            for j in range(len(source[i]["input_ids"])):
                if source[i]["input_ids"][j] != source[i]["labels"][j]:
                    source[i]["input_ids"][j] = 103
        return source

    def transpose(inputs):
        features = []
        for i in tqdm(range(len(inputs["input_ids"]))):
            #ugly fix for encoder model (the same length
            features.append({key:inputs[key][i] for key in inputs.keys()}) #we fix here (truncation 
        return features 

    return mask(transpose(train_source_tok)), mask(transpose(valid_source_tok)), mask(transpose(test_source_tok))


from fastNLP import cache_results
@cache_results(_cache_fp='cache/sighan_mask', _refresh=True)
def load_sighan_mask(path_head=""):
    """
    Tokenizer : 
        to valid mask the original wrong char of source, train bert to predict

    """
    print("[Data] Loading SigHan but Mask original wrong char Dataset ...")

    train_source_path = path_head + "./data/rawdata/sighan/std/train.src"
    train_target_path = path_head + "./data/rawdata/sighan/std/train.tgt"
    valid_source_path = path_head + "./data/rawdata/sighan/std/valid.src"
    valid_target_path = path_head + "./data/rawdata/sighan/std/valid.tgt"#valid should be same to test ( sighan 15
    test_source_path = path_head + "./data/rawdata/sighan/std/test.src"
    test_target_path = path_head + "./data/rawdata/sighan/std/test.tgt"

    train_source = wash_n(read_csv(train_source_path))#[:2000]#[274144:]#for only sighan
    train_target = wash_n(read_csv(train_target_path))#[:2000]#[274144:]
    
    valid_source = wash_n(read_csv(valid_source_path))
    valid_target = wash_n(read_csv(valid_target_path))

    test_source = wash_n(read_csv(test_source_path))
    test_target = wash_n(read_csv(test_target_path))

    #valid_source = train_source[:1100]#for valid overfit problem
    #valid_target = train_target[:1100]

    tokenizer_model_name_path="hfl/chinese-roberta-wwm-ext"

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_model_name_path)

    train_source_tok = tokenizer.batch_encode_plus(train_source, return_token_type_ids=False)#seems transformers max_length not work
    train_target_tok = tokenizer.batch_encode_plus(train_target, return_token_type_ids=False)#remove padding=True, max_length=512
    valid_source_tok = tokenizer.batch_encode_plus(valid_source, return_token_type_ids=False)
    valid_target_tok = tokenizer.batch_encode_plus(valid_target, return_token_type_ids=False)
    test_source_tok = tokenizer.batch_encode_plus(test_source, return_token_type_ids=False)
    test_target_tok = tokenizer.batch_encode_plus(test_target, return_token_type_ids=False)

    train_source_tok["labels"] = train_target_tok["input_ids"]
    valid_source_tok["labels"] = valid_target_tok["input_ids"]
    test_source_tok["labels"] = test_target_tok["input_ids"]

    # a ugly mask to replace the 
    def mask(source):
        for i in range(len(source)):
            for j in range(len(source[i]["input_ids"])):
                if source[i]["input_ids"][j] != source[i]["labels"][j]:
                    source[i]["input_ids"][j] = 103
        return source

    def transpose(inputs):
        features = []
        for i in tqdm(range(len(inputs["input_ids"]))):
            #ugly fix for encoder model (the same length
            features.append({key:inputs[key][i][:128] for key in inputs.keys()}) #we fix here (truncation 
        return features 

    print("[Data] Hint: Rough code here to mask will cause program slow")

    return mask(transpose(train_source_tok)), mask(transpose(valid_source_tok)), mask(transpose(test_source_tok))

from fastNLP import cache_results
@cache_results(_cache_fp='cache/sighan_chinesebert_ReaLiSe', _refresh=False)
def load_sighan_chinesebert_ReaLiSe(path_head=""):
    """
    Tokenizer : 
        import torch
        from chinesebert import ChineseBertForMaskedLM, ChineseBertTokenizerFast, ChineseBertConfig
        pretrained_model_name = "junnyu/ChineseBERT-base"

        tokenizer = ChineseBertTokenizerFast.from_pretrained(pretrained_model_name)
        chinese_bert = ChineseBertForMaskedLM.from_pretrained(pretrained_model_name)

        text = "北京是[MASK]国的首都。"
        inputs = tokenizer(text, return_tensors="pt")

        with torch.no_grad():
            o = chinese_bert(**inputs)
            l = o.logits
    """
    print("[Data] Loading SigHan ReaLiSe ChineseBert Dataset ...")

    train_source_path = path_head + "./data/rawdata/sighan/raw/train.src"
    train_target_path = path_head + "./data/rawdata/sighan/raw/train.tgt"
    valid_source_path = path_head + "./data/rawdata/sighan/raw/valid.src"
    valid_target_path = path_head + "./data/rawdata/sighan/raw/valid.tgt"#valid should be same to test ( sighan 15
    test_source_path = path_head + "./data/rawdata/sighan/raw/test.src"
    test_target_path = path_head + "./data/rawdata/sighan/raw/test.tgt"

    train_source = read_csv(train_source_path)#[:2000]#[274144:]#for only sighan
    train_target = read_csv(train_target_path)#[:2000]#[274144:]
    
    valid_source = read_csv(valid_source_path)
    valid_target = read_csv(valid_target_path)

    test_source = read_csv(test_source_path)
    test_target = read_csv(test_target_path)

    from chinesebert import ChineseBertTokenizerFast
    pretrained_model_name = "junnyu/ChineseBERT-base"

    tokenizer = ChineseBertTokenizerFast.from_pretrained(pretrained_model_name)

    train_source_tok = tokenizer(train_source, padding=True, truncation=True, max_length=128)
    train_target_tok = tokenizer(train_target, padding=True, truncation=True, max_length=128) 
    valid_source_tok = tokenizer(valid_source, padding=True, truncation=True, max_length=128)
    valid_target_tok = tokenizer(valid_target, padding=True, truncation=True, max_length=128) 
    test_source_tok = tokenizer(test_source, padding=True, truncation=True, max_length=128) 
    test_target_tok = tokenizer(test_target, padding=True, truncation=True, max_length=128) 

    train_source_tok["labels"] = train_target_tok["input_ids"]
    valid_source_tok["labels"] = valid_target_tok["input_ids"]
    test_source_tok["labels"] = test_target_tok["input_ids"]

    def transpose(inputs):
        features = []
        for i in tqdm(range(len(inputs["input_ids"]))):
            #ugly fix for encoder model (the same length
            features.append({key:inputs[key][i] for key in inputs.keys()}) 
        return features 

    return transpose(train_source_tok), transpose(valid_source_tok), transpose(test_source_tok)

from fastNLP import cache_results
@cache_results(_cache_fp='cache/sighan_holy', _refresh=False)
def load_sighan_holy(path_head=""):
    """
        "Holy" means remvoe all the overlapped pair between train and test
    """
    print("[Data] Loading SigHan Holy Dataset ...")

    train_source_path = path_head + "./data/rawdata/sighan/holy/train.src"
    train_target_path = path_head + "./data/rawdata/sighan/holy/train.tgt"
    valid_source_path = path_head + "./data/rawdata/sighan/holy/valid.src"
    valid_target_path = path_head + "./data/rawdata/sighan/holy/valid.tgt"#valid should be same to test ( sighan 15
    test_source_path = path_head + "./data/rawdata/sighan/holy/test.src"
    test_target_path = path_head + "./data/rawdata/sighan/holy/test.tgt"

    train_source = read_csv(train_source_path)#[:2000]#[274144:]#for only sighan
    train_target = read_csv(train_target_path)#[:2000]#[274144:]
    
    valid_source = read_csv(valid_source_path)
    valid_target = read_csv(valid_target_path)

    test_source = read_csv(test_source_path)
    test_target = read_csv(test_target_path)

    tokenizer_model_name_path="hfl/chinese-roberta-wwm-ext"

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_model_name_path)

    train_source_tok = tokenizer.batch_encode_plus(train_source, return_token_type_ids=False)#seems transformers max_length not work
    train_target_tok = tokenizer.batch_encode_plus(train_target, return_token_type_ids=False)#remove padding=True, max_length=512
    valid_source_tok = tokenizer.batch_encode_plus(valid_source, return_token_type_ids=False)
    valid_target_tok = tokenizer.batch_encode_plus(valid_target, return_token_type_ids=False)
    test_source_tok = tokenizer.batch_encode_plus(test_source, return_token_type_ids=False)
    test_target_tok = tokenizer.batch_encode_plus(test_target, return_token_type_ids=False)

    train_source_tok["labels"] = train_target_tok["input_ids"]
    valid_source_tok["labels"] = valid_target_tok["input_ids"]
    test_source_tok["labels"] = test_target_tok["input_ids"]

    def transpose(inputs):
        features = []
        for i in tqdm(range(len(inputs["input_ids"]))):
            #ugly fix for encoder model (the same length
            features.append({key:inputs[key][i][:128] for key in inputs.keys()}) #we fix here (truncation 
        return features 

    return transpose(train_source_tok), transpose(valid_source_tok), transpose(test_source_tok)


from fastNLP import cache_results
@cache_results(_cache_fp='cache/sighan_holy_mask', _refresh=True)
def load_sighan_holy_mask(path_head=""):
    """
        
        "Holy" means remvoe all the overlapped pair between train and test
    

    Tokenizer : 
        to valid mask the original wrong char of source, train bert to predict

    """
    print("[Data] Loading SigHan Holy but Mask original wrong char Dataset ...")

    train_source_path = path_head + "./data/rawdata/sighan/holy/train.src"
    train_target_path = path_head + "./data/rawdata/sighan/holy/train.tgt"
    valid_source_path = path_head + "./data/rawdata/sighan/holy/valid.src"
    valid_target_path = path_head + "./data/rawdata/sighan/holy/valid.tgt"#valid should be same to test ( sighan 15
    test_source_path = path_head + "./data/rawdata/sighan/holy/test.src"
    test_target_path = path_head + "./data/rawdata/sighan/holy/test.tgt"

    train_source = wash_n(read_csv(train_source_path))#[:2000]#[274144:]#for only sighan
    train_target = wash_n(read_csv(train_target_path))#[:2000]#[274144:]
    
    valid_source = wash_n(read_csv(valid_source_path))
    valid_target = wash_n(read_csv(valid_target_path))

    test_source = wash_n(read_csv(test_source_path))
    test_target = wash_n(read_csv(test_target_path))

    #valid_source = train_source[:1100]#for valid overfit problem
    #valid_target = train_target[:1100]

    tokenizer_model_name_path="hfl/chinese-roberta-wwm-ext"

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_model_name_path)

    train_source_tok = tokenizer.batch_encode_plus(train_source, return_token_type_ids=False)#seems transformers max_length not work
    train_target_tok = tokenizer.batch_encode_plus(train_target, return_token_type_ids=False)#remove padding=True, max_length=512
    valid_source_tok = tokenizer.batch_encode_plus(valid_source, return_token_type_ids=False)
    valid_target_tok = tokenizer.batch_encode_plus(valid_target, return_token_type_ids=False)
    test_source_tok = tokenizer.batch_encode_plus(test_source, return_token_type_ids=False)
    test_target_tok = tokenizer.batch_encode_plus(test_target, return_token_type_ids=False)

    train_source_tok["labels"] = train_target_tok["input_ids"]
    valid_source_tok["labels"] = valid_target_tok["input_ids"]
    test_source_tok["labels"] = test_target_tok["input_ids"]

    # a ugly mask to replace the 
    def mask(source):
        for i in range(len(source)):
            for j in range(len(source[i]["input_ids"])):
                if source[i]["input_ids"][j] != source[i]["labels"][j]:
                    source[i]["input_ids"][j] = 103
        return source

    def transpose(inputs):
        features = []
        for i in tqdm(range(len(inputs["input_ids"]))):
            #ugly fix for encoder model (the same length
            features.append({key:inputs[key][i][:128] for key in inputs.keys()}) #we fix here (truncation 
        return features 

    print("[Data] Hint: Rough code here to mask will cause program slow")

    return mask(transpose(train_source_tok)), mask(transpose(valid_source_tok)), mask(transpose(test_source_tok))


from fastNLP import cache_results
@cache_results(_cache_fp='cache/sighan_enchanted', _refresh=False)
def load_sighan_enchanted(path_head=""):

    print("[Data] Loading SigHan Enchanted Dataset ...")

    train_source_path = path_head + "./data/rawdata/sighan/enchanted/train.src"
    train_target_path = path_head + "./data/rawdata/sighan/enchanted/train.tgt"
    valid_source_path = path_head + "./data/rawdata/sighan/enchanted/valid.src"
    valid_target_path = path_head + "./data/rawdata/sighan/enchanted/valid.tgt"#valid should be same to test ( sighan 15
    test_source_path = path_head + "./data/rawdata/sighan/enchanted/test.src"
    test_target_path = path_head + "./data/rawdata/sighan/enchanted/test.tgt"

    train_source = read_csv(train_source_path)#[:2000]#[274144:]#for only sighan
    train_target = read_csv(train_target_path)#[:2000]#[274144:]
    
    valid_source = read_csv(valid_source_path)
    valid_target = read_csv(valid_target_path)

    test_source = read_csv(test_source_path)
    test_target = read_csv(test_target_path)

    #valid_source = train_source[:1100]#for valid overfit problem
    #valid_target = train_target[:1100]
    #train_source = train_source[1100:]
    #train_target = train_target[1100:]

    tokenizer_model_name_path="hfl/chinese-roberta-wwm-ext"

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_model_name_path)

    train_source_tok = tokenizer.batch_encode_plus(train_source, return_token_type_ids=False)#seems transformers max_length not work
    train_target_tok = tokenizer.batch_encode_plus(train_target, return_token_type_ids=False)#remove padding=True, max_length=512
    valid_source_tok = tokenizer.batch_encode_plus(valid_source, return_token_type_ids=False)
    valid_target_tok = tokenizer.batch_encode_plus(valid_target, return_token_type_ids=False)
    test_source_tok = tokenizer.batch_encode_plus(test_source, return_token_type_ids=False)
    test_target_tok = tokenizer.batch_encode_plus(test_target, return_token_type_ids=False)

    train_source_tok["labels"] = train_target_tok["input_ids"]
    valid_source_tok["labels"] = valid_target_tok["input_ids"]
    test_source_tok["labels"] = test_target_tok["input_ids"]

    def transpose(inputs):
        features = []
        for i in tqdm(range(len(inputs["input_ids"]))):
            #ugly fix for encoder model (the same length
            features.append({key:inputs[key][i][:128] for key in inputs.keys()}) #we fix here (truncation 
        return features 

    return transpose(train_source_tok), transpose(valid_source_tok), transpose(test_source_tok)

def load_sighan13_test():
    """
    UnFinished ...
    """
    print("[Data] Loading SigHan13 Test Dataset ...")

    valid_source_path = "./data/rawdata/sighan/raw/valid14.src"
    valid_target_path = "./data/rawdata/sighan/raw/valid14.tgt"

    valid_source = read_csv(valid_source_path)
    valid_target = read_csv(valid_target_path)

    tokenizer_model_name_path="hfl/chinese-roberta-wwm-ext"

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_model_name_path)

    valid_source_tok = tokenizer.batch_encode_plus(valid_source, return_token_type_ids=False)
    valid_target_tok = tokenizer.batch_encode_plus(valid_target, return_token_type_ids=False)

    valid_source_tok["labels"] = valid_target_tok["input_ids"]

    def transpose(inputs):
        features = []
        for i in tqdm(range(len(inputs["input_ids"]))):
            features.append({key:inputs[key][i][:128] for key in inputs.keys()}) #we fix here (truncation 
        return features 

    return transpose(valid_source_tok)

def load_sighan14_test(path_head="./"):
    """
    """
    print("[Data] Loading SigHan14 Test Dataset ...")

    valid_source_path = path_head + "data/rawdata/sighan/raw/valid14.src"
    valid_target_path = path_head + "data/rawdata/sighan/raw/valid14.tgt"

    valid_source = read_csv(valid_source_path)
    valid_target = read_csv(valid_target_path)

    tokenizer_model_name_path="hfl/chinese-roberta-wwm-ext"

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_model_name_path)

    valid_source_tok = tokenizer.batch_encode_plus(valid_source, return_token_type_ids=False)
    valid_target_tok = tokenizer.batch_encode_plus(valid_target, return_token_type_ids=False)

    valid_source_tok["labels"] = valid_target_tok["input_ids"]

    def transpose(inputs):
        features = []
        for i in tqdm(range(len(inputs["input_ids"]))):
            features.append({key:inputs[key][i][:128] for key in inputs.keys()}) #we fix here (truncation 
        return features 

    return transpose(valid_source_tok)

def load_sighan15_test(path_head="./"):
    """
    """
    print("[Data] Loading SigHan15 Test Dataset ...")

    valid_source_path = "./data/rawdata/sighan/std/test.src"
    valid_target_path = "./data/rawdata/sighan/std/test.tgt"

    valid_source = read_csv(valid_source_path)
    valid_target = read_csv(valid_target_path)

    tokenizer_model_name_path="hfl/chinese-roberta-wwm-ext"

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_model_name_path)

    valid_source_tok = tokenizer.batch_encode_plus(valid_source, return_token_type_ids=False)
    valid_target_tok = tokenizer.batch_encode_plus(valid_target, return_token_type_ids=False)

    valid_source_tok["labels"] = valid_target_tok["input_ids"]

    def transpose(inputs):
        features = []
        for i in tqdm(range(len(inputs["input_ids"]))):
            features.append({key:inputs[key][i][:128] for key in inputs.keys()}) #we fix here (truncation 
        return features 

    return transpose(valid_source_tok)

def split_lattice_and_source(source):
    """
    for flat
    source:
        ssss
        xxx, 123
        ssss
        xxxxx,132131
    return:
        ["sss", "sss"] , [ "xxx,123", "xxxxx, 132131" ]
    """
    res, lattice = [], []
    for i in range(0, len(source), 2):
        res.append(source[i])
        lattice.append(source[i+1])

    return res, lattice

def split_lattice_and_source_plus(source):
    """
    source:
        ssss
        xxx, 123
        ssss
        xxxxx,132131
    """
    res, lattice, sub_length = [], [], []
    for i in range(0, len(source), 2):
        magic = source[i+1].split(",")
        res.append(source[i] + "".join(magic[0].split()))
        lattice.append( list(range(len(source[i]))) + [ int(n) for s in magic[-1].split() for n in s.split("$") ] )
        sub_length.append( len(source[i]))

    return res, lattice, sub_length

def load_raw_lattice(raw_lattice_path="/data/rawdata/sighan/lattice/", path_head="."):
    """
    for flat
    dont care about para: 'path_head', only use for debugging
    """
    train_source_path = path_head + raw_lattice_path + "train.src"
    train_target_path = path_head + raw_lattice_path + "train.tgt"
    valid_source_path = path_head + raw_lattice_path + "valid.src"
    valid_target_path = path_head + raw_lattice_path + "valid.tgt"
    test_source_path = path_head + raw_lattice_path + "test.src"
    test_target_path = path_head + raw_lattice_path + "test.tgt"
    
    train_source = wash_n(read_csv(train_source_path))#[:2000]#[548288:]#[:2000]
    train_target = wash_n(read_csv(train_target_path))#[:2000]#[274144:]#[:1000]#
    
    valid_source = wash_n(read_csv(valid_source_path))
    valid_target = wash_n(read_csv(valid_target_path))

    #valid_source = train_source[:2200]#for valid overfit problem
    #valid_target = train_target[:1100]
    #train_source = train_source[2200:]
    #train_target = train_target[1100:]

    test_source = wash_n(read_csv(test_source_path))
    test_target = wash_n(read_csv(test_target_path))

    train_source, train_lattice = split_lattice_and_source(train_source)
    valid_source, valid_lattice = split_lattice_and_source(valid_source)
    test_source, test_lattice = split_lattice_and_source(test_source)

    return (train_source, train_lattice, train_target), (valid_source, valid_lattice, valid_target), (test_source, test_lattice, test_target)

def trans2dataset(source_and_lattice_and_target, max_length=100024):
    """
    for flat
    source: ["sentence", "sentence"]
    return: <class 'fastNLP.core.dataset.DataSet'>
    """    
    from fastNLP import DataSet

    source, lattice, target = source_and_lattice_and_target

    #r = "[A-Za-z0-9_.!+-=——,$%^，。？、~@#￥%……&*《》<>「」{}【】()/]"

    finals, atten_masks, lex_nums,  pos_s, pos_e, target_host = [], [], [], [], [], []

    for i in range(len(source)):

        tmp_source  = [i for i in source[i]]

        tmp_target = [i for i in target[i]]

        tmp_lattice, tmp_pos =  lattice[i].split(",")#"s s s,a a a" -> "s s s", "a a a"

        tmp_pos_s, tmp_pos_e = list(range(len(tmp_source))) + list(itertools.chain(*list(map(lambda x: [int(i) for i in x.split('$')], tmp_pos.split())))), \
                    list(range(len(tmp_source))) + list(itertools.chain(*list(map(lambda x: [int(i) for i in x.split('$')], tmp_pos.split()))))

        tmp_lattice = [u for u in "".join(tmp_lattice.split())]

        new_tmp_lattice, new_tmp_pos_s, new_tmp_pos_e = [], list(range(len(tmp_source))), list(range(len(tmp_source)))
        
        #here we mask the word left only char since we think char is a better road to answer
        for j in range(len(tmp_lattice)):
            _seq_len_ = len(tmp_source)
            index = tmp_pos_s[j + _seq_len_]

            if tmp_lattice[j] != tmp_source[index]:
                new_tmp_lattice.append(tmp_lattice[j])
                new_tmp_pos_s.append(index)
                new_tmp_pos_e.append(index)

        concated = (tmp_source + new_tmp_lattice)[:max_length]
        tmp_pos_s, tmp_pos_e = new_tmp_pos_s[:max_length], new_tmp_pos_e[:max_length]

        finals.append(concated)

        lex_nums.append(len(concated) - len(tmp_source) )

        atten_mask = [1] * len(concated)

        atten_masks.append( atten_mask[:max_length] )

        pos_s.append(tmp_pos_s[:max_length])

        pos_e.append(tmp_pos_e[:max_length])

        target_host.append(tmp_target[:max_length])


    return DataSet({ "lattice":finals, "lex_nums":lex_nums, "attention_mask":atten_masks, "target": target_host, "pos_s":pos_s, "pos_e":pos_e})#bigram just a repalced name for char

from fastNLP import cache_results
@cache_results(_cache_fp='cache/sighan_lattice_test', _refresh=True)
def load_lattice_sighan(dataset=None, path_head=""):
    """
    Temporary deprecation ！
    for flat
    #use faskNLP module because Im FDU CS ! #Im LAZY.
    """
    print("[Data] Loading Lattice SigHan Dataset ...")

    pretrain_embedding_path = "/remote-home/xtzhang/CTC/CTC2021/SpecialEdition/data/others/yangjie_word_char_mix.txt"
    
    train_pkg, valid_pkg, test_pkg = load_raw_lattice(path_head=path_head) 

    train_dataset, valid_dataset, test_dataset = trans2dataset(train_pkg), trans2dataset(valid_pkg), trans2dataset(test_pkg)

    #((train_source, train_lattice), train_target)
    datasets = dict()
    datasets["train"], datasets["valid"], datasets["test"] = train_dataset, valid_dataset, test_dataset

    datasets['valid'].add_seq_len('lattice')
    datasets['test'].add_seq_len('lattice')
    datasets['train'].add_seq_len('lattice')

    from fastNLP.core import Vocabulary
    lattice_vocab = Vocabulary()
    vocabs = dict()
    vocabs['lattice'] = lattice_vocab
    lattice_vocab.from_dataset(
        datasets['train'], 
        field_name='lattice',
        no_create_entry_dataset=[v for k, v in datasets.items() if k != 'train']
        )

    from fastNLP.embeddings import StaticEmbedding
    lattice_embedding = StaticEmbedding(
        lattice_vocab, 
        pretrain_embedding_path, word_dropout=0.01,
        min_freq=1, requires_grad=False
        )

    label_vocab = Vocabulary()
    label_vocab.from_dataset(datasets['train'], field_name='target')
    vocabs['label'] = label_vocab

    lattice_vocab.index_dataset(
        *(datasets.values()), field_name='lattice', new_field_name='input_ids'
    )
    label_vocab.index_dataset(
        *(datasets.values()), field_name='target', new_field_name='target'
    )
    #transform to transformers BatchEncoding
    #from transformers.tokenization_utils_base import BatchEncoding

    def tmp_transform(fnlp_dataset):
        new = {}
        length = None
        for key in fnlp_dataset.field_arrays:
            new[key] = [ i for i in fnlp_dataset.field_arrays[key] ]
            if not length:
                length = len(new[key])

        encodings = []

        #from tokenizers import Encoding
        for i in range(length):
            tmp = { key:fnlp_dataset.field_arrays[key][i] for key in fnlp_dataset.field_arrays }
            encodings.append(tmp)

        #res = BatchEncoding(new, encodings)
        res = myBatchEncoding(new, encodings)
        return res

    datasets = dict(zip(datasets, map(tmp_transform, datasets.values())))#

    return datasets, vocabs, lattice_embedding

def get_lattice_and_pos(source_and_lattice_and_target, tokenizer, max_length=512):
    """
    for abs pos bert
    reload version of  trans2dataset
    source: ["sentence", "sentence"]
    """    

    source, lattice, target = source_and_lattice_and_target

    finals, abs_pos, attention_masks, labels = [], [], [], []

    seq_len = []

    for i in range(len(source)):
        tmp_source = tokenizer.convert_tokens_to_ids([i for i in source[i]])#["input_ids"]

        tmp_lattice, tmp_pos =  lattice[i].split(",")#"s s s,a a a" -> "s s s", "a a a"

        tmp_pos_s  = list(range(len(tmp_source))) + list(itertools.chain(*list(map(lambda x: [int(i) for i in x.split('$')], tmp_pos.split()))))

        tmp_lattice = tokenizer.convert_tokens_to_ids([ i for i in "".join(tmp_lattice.split())])#["input_ids"]

        new_tmp_lattice, new_tmp_pos = [], list(range(len(tmp_source)))

        #here we mask the word left only char since we think char is a better road to answer
        for j in range(len(tmp_lattice)):
            _seq_len_ = len(tmp_source)
            index = tmp_pos_s[j + _seq_len_]

            if tmp_lattice[j] != tmp_source[index]:
                new_tmp_lattice.append(tmp_lattice[j])
                new_tmp_pos.append(index)#for <SOS>

        seq_len.append(len(tmp_source))
        concated = tmp_source + new_tmp_lattice[:max_length]
        tmp_pos_s= new_tmp_pos[:max_length]

        tmp_label = tokenizer.convert_tokens_to_ids([ i for i in target[i]]) 

        finals.append(concated)
        abs_pos.append(tmp_pos_s)
        attention_masks.append([1] * len(concated))
        labels.append(tmp_label)

        #expand
        #finals.append(tmp_source)
        #abs_pos.append( list(range(len(tmp_source))) )
        #attention_masks.append( [1] * len(tmp_source) )
        #labels.append(tmp_label)
        #seq_len.append(len(tmp_source))

    return {"input_ids":finals, "lattice":abs_pos, "attention_mask":attention_masks, "labels":labels, "sub_length":seq_len}

from fastNLP import cache_results
@cache_results(_cache_fp='cache/sighan_abs_pos_test', _refresh=True)
def load_abs_pos_sighan(dataset=None, path_head=""):
    """
    Temporary deprecation ！
    for abs pos bert
    """

    print("[Data] Loading Abs_Pos Bert SigHan Dataset ...")

    train_pkg, valid_pkg, test_pkg = load_raw_lattice(path_head=path_head) 

    tokenizer_model_name_path="hfl/chinese-roberta-wwm-ext"

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_model_name_path)

    train_dataset, valid_dataset, test_dataset = get_lattice_and_pos(train_pkg, tokenizer), get_lattice_and_pos(valid_pkg, tokenizer), get_lattice_and_pos(test_pkg, tokenizer)

    def transpose(inputs):
        features = []
        for i in tqdm(range(len(inputs["input_ids"]))):
            #ugly fix for encoder model (the same length
            features.append({key:inputs[key][i] for key in inputs.keys()}) #we fix here (truncation 
        return features 

    return transpose(train_dataset), transpose(valid_dataset), transpose(test_dataset)

from fastNLP import cache_results
@cache_results(_cache_fp='cache/sighan_abs_pos_plus', _refresh=True)
def load_abs_pos_sighan_plus(dataset=None, path_head=""):
    """
    Temporary deprecation ！
    for abs pos bert
    """

    print("[Data] Loading Expanded Abs_Pos Bert SigHan Dataset ...")

    train_pkg, valid_pkg, test_pkg = load_raw_lattice(path_head=path_head) 

    tokenizer_model_name_path="hfl/chinese-roberta-wwm-ext"

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_model_name_path)

    train_dataset, valid_dataset, test_dataset = get_lattice_and_pos_plus(train_pkg, tokenizer), get_lattice_and_pos(valid_pkg, tokenizer), get_lattice_and_pos(test_pkg, tokenizer)

    def transpose(inputs):
        features = []
        for i in tqdm(range(len(inputs["input_ids"]))):
            #ugly fix for encoder model (the same length
            features.append({key:inputs[key][i] for key in inputs.keys()}) #we fix here (truncation 
        return features 

    return transpose(train_dataset), transpose(valid_dataset), transpose(test_dataset)

def get_lattice_and_pos_and_spe_token(source_and_lattice_and_target, tokenizer, max_length=512):
    """
    for abs pos bert
    replace the same char with <RAW>
    source: ["sentence", "sentence"]
    """    

    source, lattice, target = source_and_lattice_and_target

    finals, abs_pos, attention_masks, labels = [], [], [], []

    seq_len = []

    for i in range(len(source)):

        tmp_source = tokenizer.convert_tokens_to_ids([i for i in source[i]])#["input_ids"]

        tmp_lattice, tmp_pos =  lattice[i].split(",")#"s s s,a a a" -> "s s s", "a a a"

        tmp_pos_s  = list(range(len(tmp_source))) + list(itertools.chain(*list(map(lambda x: [int(i) for i in x.split('$')], tmp_pos.split()))))

        tmp_lattice = tokenizer.convert_tokens_to_ids([ i for i in "".join(tmp_lattice.split())])#["input_ids"]

        new_tmp_lattice, new_tmp_pos = [], list(range(len(tmp_source)))

        #here we mask the word left only char since we think char is a better road to answer
        for j in range(len(tmp_lattice)):
            _seq_len_ = len(tmp_source)
            #print(j, _seq_len_)
            index = tmp_pos_s[j + _seq_len_]

            if tmp_lattice[j] != tmp_source[index]:
                new_tmp_lattice.append(tmp_lattice[j])
                new_tmp_pos.append(index)#for <SOS>

        seq_len.append(len(tmp_source))
        concated = tmp_source + new_tmp_lattice[:max_length]
        tmp_pos_s= new_tmp_pos[:max_length]

        label = []

        for j in range(len(target[i])):
            if target[i][j] == source[i][j]:
                label.append("<RAW>")
            else:
                label.append(target[i][j])

        tmp_label = tokenizer.convert_tokens_to_ids(label)

        finals.append(concated)
        abs_pos.append(tmp_pos_s)
        attention_masks.append([1] * len(concated))
        labels.append(tmp_label)

    return {"input_ids":finals, "lattice":abs_pos, "attention_mask":attention_masks, "labels":labels, "sub_length":seq_len} 

from fastNLP import cache_results
@cache_results(_cache_fp='cache/sighan_abs_pos_expo_token', _refresh=True)
def load_abs_pos_and_spe_token_sighan(dataset=None, path_head=""):
    """
    Temporary deprecation ！
    for abs pos bert
    """

    print("[Data] Loading Abs_Pos and Special Token Bert SigHan Dataset ...")

    train_pkg, valid_pkg, test_pkg = load_raw_lattice(raw_lattice_path="/data/rawdata/sighan/lattice_balanced/", path_head=path_head) 

    tokenizer_model_name_path="hfl/chinese-roberta-wwm-ext"

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_model_name_path)

    tokenizer.add_special_tokens({'additional_special_tokens':["<RAW>"]})#lets replace no mod  with <RAW>

    train_dataset, valid_dataset, test_dataset = get_lattice_and_pos_and_spe_token(train_pkg, tokenizer), get_lattice_and_pos_and_spe_token(valid_pkg, tokenizer), get_lattice_and_pos_and_spe_token(test_pkg, tokenizer)

    def transpose(inputs):
        features = []
        for i in tqdm(range(len(inputs["input_ids"]))):
            #ugly fix for encoder model (the same length
            features.append({key:inputs[key][i] for key in inputs.keys()}) #we fix here (truncation 
        return features 

    return transpose(train_dataset), transpose(valid_dataset), transpose(test_dataset), tokenizer

def get_lattice_and_pos_plus(source_and_lattice_and_target, tokenizer, max_length=512):
    """
    for abs pos bert
    add the <SOS> <EOS>
    not finished
    source: ["sentence", "sentence"]
    """    

    source, lattice, target = source_and_lattice_and_target

    finals, abs_pos, attention_masks, labels = [], [], [], []

    seq_len = []

    for i in range(len(source)):
        tmp_source = tokenizer.convert_tokens_to_ids( ["[CLS]"] + [o for o in source[i]] + ["[SEP]"] )#["input_ids"]

        tmp_lattice, tmp_pos =  lattice[i].split(",")#"s s s,a a a" -> "s s s", "a a a"

        tmp_pos_s  = list(range(len(tmp_source))) + list(itertools.chain(*list(map(lambda x: [int(i) for i in x.split('$')], tmp_pos.split()))))

        tmp_lattice = tokenizer.convert_tokens_to_ids([ i for i in "".join(tmp_lattice.split())])#["input_ids"]

        new_tmp_lattice, new_tmp_pos = [], list(range(len(tmp_source)))

        #here we mask the word left only char since we think char is a better road to answer
        for j in range(len(tmp_lattice)):
            _seq_len_ = len(tmp_source)
            index = tmp_pos_s[j + _seq_len_]

            if tmp_lattice[j] != tmp_source[index]:
                new_tmp_lattice.append(tmp_lattice[j])
                new_tmp_pos.append(index+1)#for <SOS>

        seq_len.append(len(tmp_source[:max_length]))
        concated = (tmp_source + new_tmp_lattice)
        tmp_pos_s= new_tmp_pos

        tmp_label = tokenizer.convert_tokens_to_ids( ["[CLS]"] + [ i for i in target[i]] + ["[SEP]"] )  

        finals.append(concated[:max_length])
        abs_pos.append(tmp_pos_s[:max_length])
        attention_masks.append( ([1] * len(concated) )[:max_length] )
        labels.append(tmp_label[:max_length])

        #expand
        #finals.append(tmp_source)
        #abs_pos.append( list(range(len(tmp_source))) )
        #attention_masks.append( [1] * len(tmp_source) )
        #labels.append(tmp_label)
        #seq_len.append(len(tmp_source))

    return {"input_ids":finals, "lattice":abs_pos, "attention_mask":attention_masks, "labels":labels, "sub_length":seq_len}

def load_raw_lattice_lang8(raw_lattice_path="/data/rawdata/sighan/lattice_lang8/", path_head="."):
    """
    for flat
    dont care about para: 'path_head', only use for debugging
    """
    train_source_path = path_head + raw_lattice_path + "train.src"
    train_target_path = path_head + raw_lattice_path + "train.tgt"
    valid_source_path = path_head + raw_lattice_path + "valid.src"
    valid_target_path = path_head + raw_lattice_path + "valid.tgt"
    test_source_path = path_head + raw_lattice_path + "test.src"
    test_target_path = path_head + raw_lattice_path + "test.tgt"

    def wash_n(data):
        import re
        return [re.sub("\n", "", i) for i in data ]
    
    train_source = wash_n(read_csv(train_source_path))#[:2000]
    train_target = wash_n(read_csv(train_target_path))#[:2000]
    
    valid_source = wash_n(read_csv(valid_source_path))
    valid_target = wash_n(read_csv(valid_target_path))

    test_source = wash_n(read_csv(test_source_path))
    test_target = wash_n(read_csv(test_target_path))

    train_source, train_lattice = split_lattice_and_source(train_source)
    valid_source, valid_lattice = split_lattice_and_source(valid_source)
    test_source, test_lattice = split_lattice_and_source(test_source)

    return (train_source, train_lattice, train_target), (valid_source, valid_lattice, valid_target), (test_source, test_lattice, test_target)

from fastNLP import cache_results
@cache_results(_cache_fp='cache/sighan_lang8_abs_pos_test', _refresh=True)
def load_abs_pos_sighan_lang8(dataset=None, path_head=""):
    """
    Temporary deprecation ！
    for abs pos bert
    """

    print("[Data] Loading Abs_Pos Bert SigHan Dataset ...")

    train_pkg, valid_pkg, test_pkg = load_raw_lattice_lang8(path_head=path_head) 

    tokenizer_model_name_path="hfl/chinese-roberta-wwm-ext"

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_model_name_path)

    train_dataset, valid_dataset, test_dataset = get_lattice_and_pos(train_pkg, tokenizer), get_lattice_and_pos(valid_pkg, tokenizer), get_lattice_and_pos(test_pkg, tokenizer)

    def transpose(inputs):
        features = []
        for i in tqdm(range(len(inputs["input_ids"]))):
            #ugly fix for encoder model (the same length
            features.append({key:inputs[key][i] for key in inputs.keys()}) #we fix here (truncation 
        return features 

    return transpose(train_dataset), transpose(valid_dataset), transpose(test_dataset)

class myBatchEncoding():
    """
    Only Using when Debuging FaskNLP Dataset -> HuggingFace Transformers BatchEncodings
    """
    def __init__(self, data:Dict, encodings:List[Dict]):
        self.data = data
        self.encodings = encodings

    def __getitem__(self, key):
        #
        if isinstance(key, str):
            return self.data[key]
        if isinstance(key, int):
            return self.encodings[key]
        else:
            raise KeyError("Wrong Key !")

    def __len__(self):
        return len(self.encodings)


if __name__ == "__main__":
    #load_sighan()
    #a, b, c = load_lattice_sighan()
    #a,b,c = load_abs_pos_sighan_plus(path_head=".")
    """
    Check length for csc task
    """
    #a,b,c = load_sighan_enchanted()
    a, b, c = load_sighan()
    
    for index, i in enumerate(a):
        if (len(i["input_ids"]) != len(i["labels"])) or ( len(i["input_ids"]) != len(i["attention_mask"]) ):
            print(index)
            print(len(i['input_ids']))
            print(len(i['labels']))
            print(len(i['attention_mask']))
            print(i['input_ids'])
            print(i["labels"])
            print("[Data] something goes wrong!")
            exit()
    else:
        print("[Data] Seems working well !")
