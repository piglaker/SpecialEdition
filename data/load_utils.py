
import os
import sys
import json


def load_json(fp):
    f = open(fp, "r", encoding="utf-8")
    data = json.load(f)
    f.close()
    return data

def get_sighan_from_json():

    all_data = {
        "train":None,
        "dev":None,
        "test":None,
        "test14":None,
        "test15":None,
    }
    data_dir = "./data/rawdata/sighan/csc/"

    train_file1 = os.path.join(data_dir, "train_dev.json")
    train_file2 = os.path.join(data_dir, "train131415.json") 
    test14_file = os.path.join(data_dir, "test14.json")
    test15_file = os.path.join(data_dir, "test15.json")

    all_data["train"] = load_json(train_file1)
    all_data["train"].extend(load_json(train_file2))

    all_data["train"] = all_data["train"]

    #all_data["dev"] = load_json(test14_file)
    all_data["test"] = load_json(test14_file)
    all_data["test"].extend(load_json(test15_file))

    return all_data


def json2list(data):
    source, target = [], []

    for element in data:
        source.append(element["original_text"])
        target.append(element["correct_text"])

    return source, target 


def get_pos(stange_inputs):
    """
    inputs: 
    """
    pos_s, pos_e = [], []


    return pos_s, pos_e


