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
    from tqdm import tqdm

    names = os.listdir(path) 

    total_count = 0

    for name in tqdm(names):
        sub_path = path + "/" + name
        
        sub_names = os.listdir(sub_path)
        step = 0
        for sub_name in sub_names:
            tmp = read_csv(sub_path + "/" + sub_name)
            if tmp :

                tmp = re.sub(" |\n", "。", "".join(tmp))
                data = cc.convert(tmp)
                data = re.sub('[a-zA-Z0-9’!"#$%&\'（）「」()*+-/:;<=>?@?★、…【】《》？“”‘’！[\\]^_`{|}~\s]+', "", data)
                data = re.split('[。,.， \n]', data)
                
                for sentence in data:
                    total_count += len(sentence)

            step += 1

    print(total_count)

    return 

main()


print( "*"*10 + "curtain" + "*"*10)


