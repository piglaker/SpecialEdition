
import re
import json
from typing import Dict, List

from tqdm import tqdm
from transformers import (
    AutoTokenizer,
)
import jieba

import sys
sys.path.append("/remote-home/xtzhang/CTC/CTC2021/SpecialEdition")


from utils.io import read_csv

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

def load_ctc2021():

    #print("#"*5+ " Loading toy datasets for debugging ... " + '#'*5)

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

    tokenizer_model_name_path="hfl/chinese-roberta-wwm-ext"

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_model_name_path)

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


def load_sighan(path_head=""):

    print("Loading SigHan Dataset ...")

    train_source_path = path_head + "./data/rawdata/sighan/std/train.src"
    train_target_path = path_head + "./data/rawdata/sighan/std/train.tgt"
    valid_source_path = path_head + "./data/rawdata/sighan/std/valid.src"
    valid_target_path = path_head + "./data/rawdata/sighan/std/valid.tgt"
    test_source_path = path_head + "./data/rawdata/sighan/std/test.src"
    test_target_path = path_head + "./data/rawdata/sighan/std/test.tgt"

    train_source = read_csv(train_source_path)
    train_target = read_csv(train_target_path)
    
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


def load_sighan14_test():
    print("Loading SigHan14 Test Dataset ...")

    valid_source_path = "./data/rawdata/sighan/std/valid14.src"
    valid_target_path = "./data/rawdata/sighan/std/valid14.tgt"

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


def load_sighan15_test():
    print("Loading SigHan15 Test Dataset ...")

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


def load_raw_lattice(raw_lattice_path="./data/rawdata/sighan/lattice/", path_head=""):
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
    train_target = wash_n(read_csv(train_target_path))#[:1000]#
    
    valid_source = wash_n(read_csv(valid_source_path))
    valid_target = wash_n(read_csv(valid_target_path))

    test_source = wash_n(read_csv(test_source_path))
    test_target = wash_n(read_csv(test_target_path))

    train_source, train_lattice = split_lattice_and_source(train_source)
    valid_source, valid_lattice = split_lattice_and_source(valid_source)
    test_source, test_lattice = split_lattice_and_source(test_source)

    return (train_source, train_lattice, train_target), (valid_source, valid_lattice, valid_target), (test_source, test_lattice, test_target)


def trans2dataset(source_and_lattice_and_target, max_length=1000000):
    """
    for flat
    source: ["sentence", "sentence"]
    return: <class 'fastNLP.core.dataset.DataSet'>
    """    
    
    from fastNLP import DataSet

    source, lattice, target = source_and_lattice_and_target

    #r = "[A-Za-z0-9_.!+-=——,$%^，。？、~@#￥%……&*《》<>「」{}【】()/]"
    r = ""

    def f(sentences):
        """
        input: [sentence, sentence]
        return: [ [s,e,n,t,e,n,c,e], ... ]
        """
        import jieba
        #return  list(map(lambda x:[x[j:j+2] for j in range(len(x) -1)][:max_length], sentences)) 
        return list(map(lambda x:list(jieba.cut(x)), sentences))
        #return list(map(lambda x:[i for i in re.sub(r, "", x)[:max_length]], sentences))

    finals, atten_masks, lex_nums,  pos_s, pos_e, target_host = [], [], [], [], [], []

    unmatch_washed_count = 0

    for i in range(len(source)):

        tmp_source  = [i for i in source[i]]

        #tmp_source = [ source[j:j+2] for j in range(len(source[i]) -1) ]

        #tmp_source = list(jieba.cut(source[i]))

        tmp_target = [i for i in target[i]]

        #tmp_target = list(jieba.cut(target[i]))

        #tmp_target = []
        #pointer = 0
        #for sub in tmp_source:
        #    tmp_target.append(target[i][pointer:pointer+len(sub)])
        #    pointer += len(sub)

        #if len(tmp_source) != len(tmp_target):
        #    print(tmp_source)
        #    print(tmp_target)
        #    unmatch_washed_count += 1
        #    continue

        #print(tmp_source)
        #print(tmp_target)
        tmp_lattice, tmp_pos =  lattice[i].split(",")#"s s s,a a a" -> "s s s", "a a a"

        concated = (tmp_source + tmp_lattice.split())[:max_length]

        finals.append(concated)

        lex_nums.append(len(concated) - len(tmp_source) )

        atten_mask = [1] * len(concated)

        atten_masks.append( atten_mask[:max_length] )

        tmp_pos_s, tmp_pos_e = list(range(len(tmp_source))) + list(map(lambda x: int(x.split('$')[0]), tmp_pos.split())), \
                    list(range(len(tmp_source))) + list(map(lambda x: int(x.split('$')[-1]), tmp_pos.split()))

        pos_s.append(tmp_pos_s[:max_length])

        pos_e.append(tmp_pos_e[:max_length])

        target_host.append(tmp_target[:max_length])

    #print("unmatch_washed_count: ", unmatch_washed_count)

    #exit()

    return DataSet({ "lattice":finals, "lex_nums":lex_nums, "attention_mask":atten_masks, "target": target_host, "pos_s":pos_s, "pos_e":pos_e})#bigram just a repalced name for char


from fastNLP import cache_results
@cache_results(_cache_fp='cache/sighan_lattice_test', _refresh=True)
def load_lattice_sighan(dataset=None, path_head=""):
    """
    Temporary deprecation ！
    for flat
    #use faskNLP module because Im FDU CS ! #Im LAZY.
    """

    print("Loading Lattice SigHan Dataset ...")

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


class myBatchEncoding():
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
    a, b, c = load_lattice_sighan()
    print(a[0])
    for i in b:
        print(i)
    print("Seems working well !")
