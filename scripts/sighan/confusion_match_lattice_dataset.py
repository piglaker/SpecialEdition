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

    train_source, train_target, train_ids = json2list(data["train"], False)

    valid14_source, valid14_target, valid14_ids = json2list(data["valid14"], False)

    valid_source, valid_target, valid_ids = json2list(data["valid"], False)

    all_train = train_source + valid14_source + valid_source

    all_target = train_target + valid14_target + valid_target

    all_ids = train_ids + valid14_ids + valid_ids
    
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
            #tmp = train_source[i] + "\n" + " ".join(result[i][0]) + "," + " ".join(result[i][1])
            #print(train_source[i].split(""))
            #lexions = [ lexion for lexion in super_get(re.sub("[a-zA-Z0-9\W*]", "", train_source[i])) if lexion[-1] in word_dict]
            lexions = [ lexion for lexion in super_get(train_source[i]) if lexion[-1] in word_dict] 
            tmp = train_source[i] + "\n" + \
                " ".join([ lexion[-1] for lexion in lexions]) + "," + \
                    " ".join( [ "$".join(list(map(str, list(range(lexion[0], lexion[1]+1))))) for lexion in lexions] )
                
            #for lexion in lexions:
            #    print(list(range(lexion[0], lexion[1]+1)))
            #print( " ".join( [ "$".join(list(map(str, list(range(lexion[0], lexion[1]))))) for lexion in lexions] ) )
            #print(tmp) 
            new.append(tmp)
            #exit()
        return new

    train_magic_source, valid14_magic_source, valid_magic_source = app(train_source), app(valid14_source), app(valid_source)   
 
    write_to("/remote-home/xtzhang/CTC/CTC2021/SpecialEdition/data/rawdata/sighan/lattice/train.src", "\n".join(train_magic_source))
    write_to("/remote-home/xtzhang/CTC/CTC2021/SpecialEdition/data/rawdata/sighan/lattice/train.tgt", "\n".join(train_target))

    write_to("/remote-home/xtzhang/CTC/CTC2021/SpecialEdition/data/rawdata/sighan/lattice/valid14.src", "\n".join(valid14_magic_source))
    write_to("/remote-home/xtzhang/CTC/CTC2021/SpecialEdition/data/rawdata/sighan/lattice/valid14.tgt", "\n".join(valid14_target))

    write_to("/remote-home/xtzhang/CTC/CTC2021/SpecialEdition/data/rawdata/sighan/lattice/test.src", "\n".join(valid_magic_source))
    write_to("/remote-home/xtzhang/CTC/CTC2021/SpecialEdition/data/rawdata/sighan/lattice/test.tgt", "\n".join(valid_target))

    write_to("/remote-home/xtzhang/CTC/CTC2021/SpecialEdition/data/rawdata/sighan/lattice/valid.src", "\n".join(valid_magic_source[500:]))
    write_to("/remote-home/xtzhang/CTC/CTC2021/SpecialEdition/data/rawdata/sighan/lattice/valid.tgt", "\n".join(valid_target[500:]))


    return 


main()