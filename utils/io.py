import os
import json


def load_json(fp):
    f = open(fp, "r", encoding="utf-8")
    data = json.load(f)
    f.close()
    return data

def read_csv(path):
    result = []
    try:
        with open(path) as f:
            for line in f.readlines():
                #result.append(line.replace(" ", ""))
                result.append(line)
        return result
    except Exception as e:
        print(e)
        return 

def write_to(path, contents):
    f = open(path, "w", encoding='utf-8')
    f.write(contents)
    f.close()


def get_sighan_from_json():

    all_data = {
        "train":None,
        "dev":None,
        "test":None,
        "test14":None,
        "test15":None,
    }
    data_dir = "/remote-home/xtzhang/CTC/CTC2021/SpecialEdition/data/rawdata/sighan/csc/"

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


def test():
    """
    
    """
    
    #Do something

    return


if __name__ == "__main__":
    test()
