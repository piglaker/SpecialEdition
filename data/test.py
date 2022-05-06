

import sys
sys.path.append("/remote-home/xtzhang/CTC/CTC2021/SpecialEdition")


from utils.io import read_csv


path_head = ""

expand_source_path = path_head + "./data/nlpcc_and_hsk/train.src"
expand_target_path = path_head + "./data/nlpcc_and_hsk/train.trg"


expand_source = read_csv(expand_source_path, remove_blank=True)
expand_target = read_csv(expand_target_path, remove_blank=True)


for i in range(len(expand_source)):
    if len(expand_source[i]) != len(expand_target[i]):
        print(expand_source[i])
        print(expand_target[i])
        exit()


