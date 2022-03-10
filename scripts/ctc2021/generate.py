
# demo:
# python generate.py v1
#
import os
import re
import sys
import json

from tqdm import tqdm

#upper import 
sys.path.append("../../") 
from utils import levenshtein

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

def preprocess(sentence):
    s = strQ2B(sentence)
    #back_num = re.findall('\d+', s)
    #back_eng = re.findall(r'[a-zA-Z]+', s)
    #s = re.sub(r'[a-zA-Z]+', 'e', s)
    #s = re.sub('\d+', 'n', s)
    return s#, back_num + back_eng

def write_to(path, contents):
    print(path)
    f = open(path, "w", encoding='utf-8')
    f.write(contents)
    f.close()

def generate(which="v2"):
    """
    split raw data(train.json) to preprocessed target
    """
    if which == "v2":
        file = open("../../data/rawdata/ctc2021/train_large_v2.json", 'r', encoding='utf-8')
    elif which == "v3":
        # shortcut
        file = open("../../data/rawdata/ctc2021/pseudo_data.json", 'r', encoding='utf-8')
        data = json.load(file)
        source, target = [], []
        for d in tqdm(data):
            source.append(preprocess(d["source"]))
            target.append(preprocess(d["target"]))

        write_to("../../data/rawdata/ctc2021/train_v3.src", "\n".join(source))
        write_to("../../data/rawdata/ctc2021/train_v3.tgt", "\n".join(target))

        return 

    elif which == "v1":
        file = open("../../data/rawdata/ctc2021/train.json", 'r', encoding='utf-8')
    else:
        print("Error")
        exit(0)
    
    source, target = [], []

    source_back, target_back = [], []

    for line in tqdm(file.readlines()):
        tmp = json.loads(line)

        s, back = preprocess(tmp["source"])

        source.append(s)
        source_back.append(back)

        s, back = preprocess(tmp["target"])

        target.append(s)
        target_back.append(back)

    print(source[:3], target[:3])

    print(len(source), len(target))

    through = []
    
    #f_src = levenstein.tokenize(source, vocab_file_path="vocab.txt")
    
    through = levenshtein.convert_from_sentpair_through(source, target, source)

    print(through[0])

    if which == "v2":
        write_to("../../data/rawdata/ctc2021/test_v2.src", "\n".join(source[200000:]))
        write_to("../../data/rawdata/ctc2021/test_v2.tgt", "\n".join(target[200000:]))
        # write_to("../../data/rawdata/ctc2021/test"+which+".through", "\n".join(through[90000:]))

        write_to("../../data/rawdata/ctc2021/valid_v2.src", "\n".join(source[200000:]))
        write_to("../../data/rawdata/ctc2021/valid_v2.tgt", "\n".join(target[200000:]))
        # write_to("../../data/rawdata/ctc2021/valid"+which+".through", "\n".join(through[90000:]))

        write_to("../../data/rawdata/ctc2021/train_v2.src", "\n".join(source[:230000]))
        write_to("../../data/rawdata/ctc2021/train_v2.tgt", "\n".join(target[:230000]))
        # write_to("../../data/rawdata/ctc2021/train"+which+".through", "\n".join(through[:90000]))


    else:
        write_to("../../data/rawdata/ctc2021/test.src", "\n".join(source[99000:]))
        write_to("../../data/rawdata/ctc2021/test.tgt", "\n".join(target[99000:]))
        # write_to("../../data/rawdata/ctc2021/test"+which+".through", "\n".join(through[90000:]))

        write_to("../../data/rawdata/ctc2021/valid.src", "\n".join(source[99000:]))
        write_to("../../data/rawdata/ctc2021/valid.tgt", "\n".join(target[99000:]))
        # write_to("../../data/rawdata/ctc2021/valid"+which+".through", "\n".join(through[90000:]))

        write_to("../../data/rawdata/ctc2021/train.src", "\n".join(source[:99000]))
        write_to("../../data/rawdata/ctc2021/train.tgt", "\n".join(target[:99000]))
        # write_to("../../data/rawdata/ctc2021/train"+which+".through", "\n".join(through[:90000]))

if __name__ == "__main__":
    import argparse  
   
    parser = argparse.ArgumentParser()
 
    parser.add_argument("which")         
 
    args = parser.parse_args()

    parser.parse_args()
 
    generate(args.which)

