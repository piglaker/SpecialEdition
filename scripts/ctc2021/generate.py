import os
import re
import sys
import json

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

def generate():
    """
    split raw data(train.json) to preprocessed target
    """
    file = open("../../data/rawdata/ctc2021/train.json", 'r', encoding='utf-8')

    source, target = [], []

    source_back, target_back = [], []

    def preprocess(sentence):
        s = strQ2B(sentence)
        back_num = re.findall('\d+', s)
        back_eng = re.findall(r'[a-zA-Z]+', s)
        s = re.sub(r'[a-zA-Z]+', 'e', s)
        s = re.sub('\d+', 'n', s)
        return s, back_num + back_eng

    for line in file.readlines():
        tmp = json.loads(line)

        s, back = preprocess(tmp["source"])

        source.append(s)
        source_back.append(back)

        s, back = preprocess(tmp["target"])

        target.append(s)
        target_back.append(back)


    print(source[:3], target[:3])

    print(len(source), len(target))

    def write_to(path, contents):
        f = open(path, "w", encoding='utf-8')
        f.write(contents)
        f.close()

    through = []
    
    #f_src = levenstein.tokenize(source, vocab_file_path="vocab.txt")
    
    through = levenshtein.convert_from_sentpair_through(source, target, source)

    print(through[0])

    write_to("../../data/rawdata/ctc2021/test.src", "\n".join(source[:2000]))
    write_to("../../data/rawdata/ctc2021/test.tgt", "\n".join(target[:2000]))
    write_to("../../data/rawdata/ctc2021/test.through", "\n".join(through[:1000]))

    write_to("../../data/rawdata/ctc2021/valid.src", "\n".join(source[2000:2500]))
    write_to("../../data/rawdata/ctc2021/valid.tgt", "\n".join(target[2000:2500]))
    write_to("../../data/rawdata/ctc2021/valid.through", "\n".join(through[2000:2500]))

    write_to("../../data/rawdata/ctc2021/train.src", "\n".join(source[2500:]))
    write_to("../../data/rawdata/ctc2021/train.tgt", "\n".join(target[2500:]))
    write_to("../../data/rawdata/ctc2021/train.through", "\n".join(through[2500:]))


if __name__ == "__main__":
    generate()

