
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

#all_data = "".join(f)

#all_data = re.sub(" |\n", "", all_data)

#all_data = cc.convert(all_data)

#print(type(all_data))

#print(all_data[:100])

#all_data = all_data.split("。")

#print(all_data[0:10])

#print(type(all_data))

from time import time

#start_time = time()
#from fastNLP import cache_results
#@cache_results(_cache_fp='cache/__wiki_corpus__', _refresh=False)
#def joint_corpus(path):
#    from tqdm import tqdm
#    all_data = []
#    names = os.listdir(path)
#    for name in tqdm(names):
#        sub_path = path + "/" + name
#        sub_names = os.listdir(sub_path)
#        for sub_name in sub_names:
#            tmp = read_csv(sub_path + "/" + sub_name)
#            if tmp :
#                new = cc.convert(re.sub(" |\n", "", "".join(tmp)))
#                all_data += new
#    return all_data

#all_data = joint_corpus(path)

#print(time() - start_time)

#print(len(all_data))

#data_c_dict = []
#from tqdm import tqdm
#for line in tqdm(all_data):
#    for c in line:
#        if '\u4e00' <= c <= '\u9fff':
#            data_c_dict.append(c)
#print(len(data_c_dict))
#data_c_dict = list(set(data_c_dict))
#print(len(data_c_dict))
#print(data_c_dict[:10])
#write_to("all_char_dict.txt", "\n".join(data_c_dict))

#my_char_dict = read_csv("my_char_dict.txt")


def get_combines(sentence):
    #print(sentence)
    n = len(sentence)

    tmp = [ (sentence[i],sentence[j]) for i in range(n) for j in range(i+1, n)]
    #print(len(tmp))
    return tmp

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
        pair_dict = pickle.load(open("pair_dict.pickle", 'rb'))
        index = int(read_csv("wiki_maxinfo.log")[0])
        print("Load continue from:" + str(index))
    except:
        print("Init new")
        pair_dict = {}
        index = 0

    names = os.listdir(path) 

    def pretty_format(sys_getsizeofobj):
        #import sys
        mem  = sys_getsizeofobj
        if mem < 1024:
            return str(mem) + "B"
        elif mem < 1024 * 1024:
            return str(mem / 1024) + "KB"
        elif mem < 1024*1024*1024:
            return str(mem / 1024 / 1024) + "MB"
        else:
            return str(mem  / (1024) ** 3) + "GB"

    def get_mem_used():
        pid = os.getpid()
        process = psutil.Process(pid)
        info = process.memory_full_info()
        memory = info.uss/1024/1024
        return " Mem used :" + str(memory) + "MB"

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
                    combines = get_combines(re.sub("\W*", "", sentence))
                    for combine in combines:
                        if combine in pair_dict:
                            pair_dict[combine] += 1
                        else:
                            pair_dict[combine] = 1
                    del combines
                #while 1:
                #    in_ = input(state + "Input something to continue :" ) 
                #    if in_ == "c":
                #        break
                #    else:
                #        try:
                #            exec_return = exec(in_)
                # 
                #            print(exec_return)
                #        except Exception:
                #            print(Exception)
                #            continue
            
            state = str(epoch)+"/"+str(total) + " " +str(step)+"/"+str(sub) + " Length: "+str(len(pair_dict.keys())) + " Mem : " + pretty_format(sys.getsizeof(pair_dict)) + get_mem_used() + "\n"
            step += 1

        import pickle
        with open ("pair_dict.pickle", 'wb') as f:
            pickle.dump(pair_dict, f)   
        write_to("wiki_maxinfo.log", str(epoch+index))

        #pair_dict = pickle.load(open("pair_dict.pickle", 'rb'))
 
        print("Epoch " + str(epoch+index) + " , save Done")

        gc.collect()  
        epoch += 1
        print(len(pair_dict.keys()))

    return pair_dict

pair_dict = main()

print(len(pair_dict.keys()))

import pickle
with open ("pair_dict.pickle", 'wb') as f:
    pickle.dump(pair_dict, f)

print( "*"*10 + "curtain" + "*"*10)

