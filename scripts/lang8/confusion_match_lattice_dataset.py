import re
import os
import sys
import json
import time

import jieba
from tqdm import tqdm

sys.path.append("../../") 
from utils.io import read_csv, write_to, load_json

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

#
def preprocess(sentence):
    s = strQ2B(sentence)
    #back_num = re.findall('\d+', s)
    #back_eng = re.findall(r'[a-zA-Z]+', s)
    #s = re.sub(r'[a-zA-Z]+', 'e', s)
    #s = re.sub('\d+', 'n', s)
    return s

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

def preprocess_plus(sentence):
    s = strQ2B(sentence)
    #back_num = re.findall('\d+', s)
    #back_eng = re.findall(r'[a-zA-Z]+', s)
    #s = re.sub(r'[a-zA-Z]+', 'e', s)
    #s = re.sub('\d+', 'n', s)
    s = re.sub("\n", "", s)
    return s

def get_lang8_from_txt():
    data_dir = "../../data/rawdata/lang8/raw/"

    source, target = read_csv(data_dir + "train.src"), read_csv(data_dir + "train.tgt")

    new_train, new_target = list(map(preprocess_plus, source)), list(map(preprocess_plus, target))

    return new_train, new_target

def light_preprocess(sentence):
    import re
    import jieba
    return [ i for i in jieba.cut(re.sub("\W*", "", sentence)) if len(i) >= 1] 

def json2list(data, need_preprocess):
    source, target, ids = [], [], []

    for element in data:

        if need_preprocess:
            source.append(preprocess(element["original_text"]))
            target.append(preprocess(element["correct_text"]))
            ids.apoend(element["wrong_ids"])
        else:
            source.append(strQ2B((element["original_text"])))
            target.append(strQ2B((element["correct_text"])))
            ids.append(element["wrong_ids"])

    return source, target, ids

def main():
    print("load raw sighan ...")
    data = get_sighan_from_json()

    train_source, train_target, _ = json2list(data["train"], False)

    valid14_source, valid14_target, _ = json2list(data["valid14"], False)

    valid_source, valid_target, _ = json2list(data["valid"], False)

    sys.path.append("/remote-home/xtzhang/CTC/CTC2021/SpecialEdition")
    from utils.io import read_csv, write_to, load_json
    from utils.trie_utils import list2confusion_trie

    print("load confusion set ...")
    confusion_set =  read_csv("/remote-home/xtzhang/CTC/CTC2021/SpecialEdition/data/confusion_set/confusion.txt")
    confusion_dict = {}
    for line in confusion_set:
        line = line.split(":")
        confusion_dict[line[0][0]] = line[-1]

    print("load all word dict ...")
    all_word_list = read_csv("/remote-home/xtzhang/CTC/CTC2021/SpecialEdition/scripts/sighan/all_word_dict.txt")
    def wash_n(all_word_list):
        return [ re.sub("\n", "", i) for i in all_word_list]
    all_word_list = wash_n(all_word_list)
    print("build trie ... ")
    trie = list2confusion_trie(all_word_list, confusion_dict)

    def super_get(sentence):
        trie.assign(sentence)
        trie.my_get_lexion()
        return trie.result

    print("load 30w dict ...")
    path = "./30wdict_utf8.txt"
    dict_ = read_csv(path)
    word_dict = {}
    for line in tqdm(dict_):
        word_dict[re.sub("\W*", "", line)] = 0

    def app(train_source):
        new = []
        for i in tqdm(range(len(train_source))):

            lexions = [ lexion for lexion in super_get(train_source[i]) if lexion[-1] in word_dict] 

            tmp = train_source[i] + "\n" + \
                " ".join([ lexion[-1] for lexion in lexions]) + "," + \
                    " ".join( [ "$".join(list(map(str, list(range(lexion[0], lexion[1]+1))))) for lexion in lexions] )
                
            new.append(tmp)
        return new

    train_magic_source, valid14_magic_source, valid_magic_source = app(train_source), app(valid14_source), app(valid_source)   

    #add lang8
    lang8_source , lang8_target = get_lang8_from_txt()

    train_extra_source = app(lang8_source)

    write_to("/remote-home/xtzhang/CTC/CTC2021/SpecialEdition/data/rawdata/sighan/lattice_lang8/train.src", "\n".join(train_magic_source + train_extra_source))
    write_to("/remote-home/xtzhang/CTC/CTC2021/SpecialEdition/data/rawdata/sighan/lattice_lang8/train.tgt", "\n".join(train_target + lang8_target))

    write_to("/remote-home/xtzhang/CTC/CTC2021/SpecialEdition/data/rawdata/sighan/lattice_lang8/valid14.src", "\n".join(valid14_magic_source))
    write_to("/remote-home/xtzhang/CTC/CTC2021/SpecialEdition/data/rawdata/sighan/lattice_lang8/valid14.tgt", "\n".join(valid14_target))

    write_to("/remote-home/xtzhang/CTC/CTC2021/SpecialEdition/data/rawdata/sighan/lattice_lang8/test.src", "\n".join(valid_magic_source))
    write_to("/remote-home/xtzhang/CTC/CTC2021/SpecialEdition/data/rawdata/sighan/lattice_lang8/test.tgt", "\n".join(valid_target))

    write_to("/remote-home/xtzhang/CTC/CTC2021/SpecialEdition/data/rawdata/sighan/lattice_lang8/valid.src", "\n".join(valid_magic_source))
    write_to("/remote-home/xtzhang/CTC/CTC2021/SpecialEdition/data/rawdata/sighan/lattice_lang8/valid.tgt", "\n".join(valid_target))


    return 

if __name__ == "__main__":
    main()