##

import os
import sys
import re
from typing import List, Dict

from tqdm import tqdm

#utils
def read_csv(path):
    result = []

    with open(path) as f:
        for line in f.readlines():
            result.append(line)

    return result

def build_structure(throughs):
    """
    #turn ['1 2 啥'] to {('1', '2'):'啥'}
    #
    #
    """

    result = []

    for through in throughs:

        through = through.split()

        pointer = 0

        catch = {}

        #c++ style code for adapt "2 3 好 的" 
        while pointer + 3 <= len(through):

            key = (through[pointer], through[pointer+1])# key like : (1,2) 
        
            catch[key] = through[pointer+2]

            while pointer + 3 < len(through) and not through[pointer+3].isdigit():

                pointer += 1

                catch[key] += through[pointer+2]
            
            pointer += 3

        result.append(catch)

    return result

### f1 score
###
#《Introduction to SIGHAN 2015 Bake-off for Chinese Spelling Check》
# https://aclanthology.org/W15-3106.pdf
#
##

def get_score(predict, target, debug = False):
    """
    Detection 
    """
    print("*"*5, "Detection", "*"*5)

    tp, fp, fn, tn = 0, 0, 0, 0

    tp_list, fp_list, fn_list = [], [], []

    for i in tqdm(range(len(predict))):
        
        dict_pre, dict_true = predict[i], target[i]
        #print(dict_pre, dict_true)

        if not dict_true and dict_pre:
            """
            Picked but Dont Need Pick
            """
            fp += 1
            fp_list.append(i)
            continue
        
        if not dict_true and not dict_pre:
            continue

        flag = False
        for key in dict_true.keys():
            if not key in dict_pre.keys():
                fn += 1
                flag = True
                fn_list.append(i)
                break
        if flag:
            continue
         
        for key in dict_pre.keys():            
            if key not in dict_true.keys():
                """
                Picked but Dont Need Pick
                """
                fn += 1
                fn_list.append(i)
                break
        else:
            """
            Picked and Need Pick
            """
            tp += 1
            tp_list.append(i)

        """
        Unpick and Dont Need Pick
        """
        #since f1 score dont need fn, so we wont calculate it
        
    print("TP: ",tp, "FP: ", fp, "FN: ", fn,)
    
    precision = tp / (tp + fp + 1e-10)

    recall = tp / (tp + fn + 1e-10)

    F1_score = 2 * precision * recall / (precision + recall + 1e-10)

    print("Precision: ", precision, "Recall: ", recall)

    print("F1_score: ", F1_score)

    return F1_score, tp_list, fp_list, fn_list

### f1 score
###
#《Introduction to SIGHAN 2015 Bake-off for Chinese Spelling Check》
# https://aclanthology.org/W15-3106.pdf
#
##

def get_score_correction(predict, target):
    """
    Correction Level
    """
    print("*"*5, "Correction", "*"*5)

    tp, fp, fn, tn = 0, 0, 0, 0

    tp_list, fp_list, fn_list = [], [], []

    for i in tqdm(range(len(predict))):

        dict_pre, dict_true = predict[i], target[i]
        #print(dict_pre, dict_true)

        if not dict_true and dict_pre:
            """
            Picked but Dont Need Pick
            """
            fp += 1
            fp_list.append(i)
            continue
        
        if not dict_true and not dict_pre:
            continue

        flag = False
        for key in dict_true.keys():
            if not key in dict_pre.keys():
                fn += 1
                flag = True
                fn_list.append(i)
                break
        if flag:
            continue

        for key in dict_pre.keys():            
            if key not in dict_true.keys():
                """
                Picked but Dont Need Pick
                """
                fn += 1
                fn_list.append(i)
                break
            elif dict_pre[key] != dict_true[key]:
                fn += 1
                fn_list.append(i)
                break
        else:
            """
            Picked and Need Pick
            """
            tp += 1
            tp_list.append(i)
            
        #since f1 score dont need fn, so we wont calculate it
        
    print("TP: ",tp, "FP: ", fp, "FN: ", fn,)
    
    precision = tp / (tp + fp + 1e-10)

    recall = tp / (tp + fn + 1e-10)

    F1_score = 2 * precision * recall / (precision + recall + 1e-10)

    print("Precision: ", precision, "Recall: ", recall)

    print("F1_score: ", F1_score)

    return F1_score, tp_list, fp_list, fn_list

def compute(predictions_path, target_path):
    """
    """
    predictions, target = read_csv(predictions_path), read_csv(target_path)

    structured_predictions, structured_target = build_structure(predictions), build_structure(target)

    _ = get_score(structured_predictions, structured_target)

    _ = get_score_correction(structured_predictions, structured_target)

    return 

def test():
    #sighan15 test
    predictions_path = "../tmp/sighan_seq/generated_predictions.through"
    target_path = "../data/rawdata/sighan/valid.through"

    compute(predictions_path=predictions_path, target_path=target_path)

    return 

if __name__ == "__main__":
    test()



