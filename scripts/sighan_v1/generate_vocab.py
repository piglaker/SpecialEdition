import os
import re
import sys
import json

#upper import 
sys.path.append("../../") 
from utils import levenshtein
from utils.io import load_json, write_to

def strQ2B(ustring):
    """全角转半角"""
    rstring = ""
    for uchar in ustring:
        inside_code=ord(uchar)
        if inside_code == 12288:                              #全角空格直接转换            
            inside_code = 32 
        elif (inside_code >= 65281 and inside_code <= 65374): #全角字符（除空格）根据关系转化
            inside_code -= 65248

        rstring += chr(inside_code)
    return rstring

def get_sighan_from_json():

    all_data = {
        "train":None,
        #"dev":None,
        #"test":None,
        #"test14":None,
        #"test15":None,
    }
    data_dir = "../../data/rawdata/sighan/csc/"

    train_file1 = os.path.join(data_dir, "train_dev.json")
    train_file2 = os.path.join(data_dir, "train131415.json") 
    test14_file = os.path.join(data_dir, "test14.json")
    test15_file = os.path.join(data_dir, "test15.json")

    all_data["train"] = load_json(train_file1)
    all_data["train"].extend(load_json(train_file2))

    all_data["train"] = all_data["train"]

    all_data["valid14"] = load_json(test14_file)
    all_data["valid"] = load_json(test15_file)
    #all_data["test"].extend(load_json(test15_file))

    return all_data

def preprocess(sentence):
    s = strQ2B(sentence)
    back_num = re.findall('\d+', s)
    back_eng = re.findall(r'[a-zA-Z]+', s)
    #s = re.sub(r'[a-zA-Z]+', 'e', s)
    #s = re.sub('\d+', 'n', s)
    return s


def json2list(data, need_preprocess):
    source, target = [], []

    for element in data:

        if need_preprocess:
            source.append(preprocess(element["original_text"]))
            target.append(preprocess(element["correct_text"]))
        else:
            source.append(strQ2B((element["original_text"])))
            target.append(strQ2B((element["correct_text"])))

    return source, target 

def generate_vocab(need_preprocess=True):
    """
    split raw data(train.json) to preprocessed target
    """
    #file = open("../../data/rawdata/ctc2021/train.json", 'r', encoding='utf-8')
    def merge(a):
        """
        """
        res = []
        for key in a.keys():
            res += a[key]
        return res

    all_data = merge(get_sighan_from_json())

    tag_vocab = {} # 1: -> 是, 2: -> 生

    tag_vocab["$KEEP"] = 0

    for data in all_data:
        for i, element in enumerate(data["original_text"]):
            if element != data["correct_text"][i]:
                tag_vocab[data["correct_text"][i]] = 0
    
    with open("vocabs.txt", "w", encoding="utf-8") as f:
        for tag in tag_vocab.keys():
            f.write(tag+"\n")
    return 
    
    
if __name__ == "__main__":
    generate_vocab()

