import os
import re
import sys
import json

#upper import 
sys.path.append("../../") 
from utils import levenshtein
from utils.io import load_json, write_to,read_csv

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

    all_data = {}
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

def generate(need_preprocess=True):
    """
    split raw data(train.json) to preprocessed target
    """

    data = get_sighan_from_json()

    vocabs = read_csv("vocabs.txt")
    import re
    char2ids = { re.sub("\n", "", vocabs[i]):str(i) for i in range(len(vocabs)) }

    def seq2tag(source_target):
        source, target = source_target
        new_target = []
        for i in range(len(source)):
            birth = [] # ["1"]
            for j, element in enumerate(source[i]):
                
                if element == target[i][j]:
                    birth.append('1')
                else:
                    birth.append(str(int(char2ids[target[i][j]])+1))
     
            #birth.append("1")

            new_target.append(",".join(birth))

        return source, new_target

    train_source, train_target = seq2tag(json2list(data["train"], need_preprocess))

    valid14_source, valid14_target = seq2tag(json2list(data["valid14"], need_preprocess))

    valid_source, valid_target = seq2tag(json2list(data["valid"], need_preprocess))


    write_to("../../data/rawdata/sighan/gector/train.src", "\n".join(train_source))
    write_to("../../data/rawdata/sighan/gector/train.tgt", "\n".join(train_target))

    write_to("../../data/rawdata/sighan/gector/valid14.src", "\n".join(valid14_source))
    write_to("../../data/rawdata/sighan/gector/valid14.tgt", "\n".join(valid14_target))

    write_to("../../data/rawdata/sighan/gector/test.src", "\n".join(valid_source))
    write_to("../../data/rawdata/sighan/gector/test.tgt", "\n".join(valid_target))

    write_to("../../data/rawdata/sighan/gector/valid.src", "\n".join(valid_source))
    write_to("../../data/rawdata/sighan/gector/valid.tgt", "\n".join(valid_target))


if __name__ == "__main__":
    generate()

