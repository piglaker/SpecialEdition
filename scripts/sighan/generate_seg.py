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
        "dev":None,
        "test":None,
        "test14":None,
        "test15":None,
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
    s = re.sub(r'[a-zA-Z]+', 'e', s)
    s = re.sub('\d+', 'n', s)
    return s

def json2list(data):
    source, target = [], []

    for element in data:
        source.append(preprocess(element["original_text"]))
        target.append(preprocess(element["correct_text"]))

    return source, target 

def generate():
    """
    split raw data(train.json) to preprocessed target
    """
    #file = open("../../data/rawdata/ctc2021/train.json", 'r', encoding='utf-8')

    data = get_sighan_from_json()

    train_source, train_target = json2list(data["train"])

    valid14_source, valid14_target = json2list(data["valid14"])

    valid_source, valid_target = json2list(data["valid"])

    print(train_source[:3], train_target[:3])
    print(len(train_source), len(train_target))
    print(valid_source[:3], valid_target[:3])
    print(len(valid_source), len(valid_target)) 


    """
    not finished yet

    write_to("../../data/rawdata/sighan/train_seg.src", "\n".join(train_source_seg))
    write_to("../../data/rawdata/sighan/train_seg.tgt", "\n".join(train_target_seg))

    write_to("../../data/rawdata/sighan/valid14_seg.src", "\n".join(valid14_source_seg))
    write_to("../../data/rawdata/sighan/valid14.tgt", "\n".join(valid14_target_seg))

    write_to("../../data/rawdata/sighan/test_seg.src", "\n".join(valid_source_seg))
    write_to("../../data/rawdata/sighan/test_seg.tgt", "\n".join(valid_target_seg))

    write_to("../../data/rawdata/sighan/valid_seg.src", "\n".join(valid_source_seg[:500]))
    write_to("../../data/rawdata/sighan/valid_seg.tgt", "\n".join(valid_target_seg[:500]))
    """

    return


def jieba_wordsegment(list_of_sentence):


    return 


if __name__ == "__main__":
    generate()

