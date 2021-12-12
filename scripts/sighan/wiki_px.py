import re
import os
import sys


path = "/remote-home/xtzhang/CTC/CTC2021/SpecialEdition/data/cn_corpus/wikiextractor/wikiextractor/extracted"

#name = "/AA/wiki_01"

sys.path.append("/remote-home/xtzhang/CTC/CTC2021/SpecialEdition")
from utils.io import read_csv, write_to

#f = read_csv(path + name)

import opencc
cc = opencc.OpenCC('t2s')

def main():
    from tqdm import tqdm
    import sys
    import gc
    import os
    import psutil
    import pickle

    #pair_dict = pickle.load(open("pair_dict.pickle", 'rb'))

    #exit()
    try:
        pair_dict = pickle.load(open("single_dict.pickle", 'rb'))
        index = int(read_csv("wiki_px.log")[0])
        print("Load continue from:" + str(index))
    except:
        print("Init new")
        pair_dict = {}
        index = 0

    names = os.listdir(path) 

    total = len(names)
    epoch = 0
    for name in tqdm(names[index+1:]):
        sub_path = path + "/" + name
        #print(sub_path)
        sub_names = os.listdir(sub_path)
        step = 0
        sub = len(sub_names)
        for sub_name in sub_names:
            max_length = 0
            max_sentence = None
            tmp = read_csv(sub_path + "/" + sub_name)
            if tmp :

                tmp = re.sub(" |\n", "。", "".join(tmp))
                data = cc.convert(tmp)
                data = re.sub('[a-zA-Z0-9’!"#$%&\'（）「」()*+-/:;<=>?@?★、…【】《》？“”‘’！[\\]^_`{|}~\s]+', "", data)
                data = re.split('[。,.， \n]', data)
                
                for sentence in data:
                    if len(sentence) > max_length:
                        max_length = len(sentence)
                        max_sentence = sentence

                    combines = [c for c in re.sub("\W*", "", sentence) ]

                    for combine in combines:
                        if combine in pair_dict:
                            pair_dict[combine] += 1
                        else:
                            pair_dict[combine] = 1
                    del combines

            step += 1
        #exit()
        import pickle
        with open ("single_dict.pickle", 'wb') as f:
            pickle.dump(pair_dict, f)   
        write_to("wiki_px.log", str(epoch+index))

        #pair_dict = pickle.load(open("single.pickle", 'rb'))
 
        print("Epoch " + str(epoch+index) + " , save Done")

        gc.collect()  
        epoch += 1
        print(len(pair_dict.keys()))

    return pair_dict

pair_dict = main()

print(len(pair_dict.keys()))

import pickle
with open ("single_dict.pickle", 'wb') as f:
    pickle.dump(pair_dict, f) 

print( "*"*10 + "curtain" + "*"*10)


