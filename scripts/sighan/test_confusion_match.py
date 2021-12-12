
import re
import os
import sys
import json
import time

import jieba

sys.path.append("/remote-home/xtzhang/CTC/CTC2021/SpecialEdition")

from utils.io import read_csv, write_to, load_json

from utils.trie_utils import list2confusion_trie


confusion_set =  read_csv("/remote-home/xtzhang/CTC/CTC2021/SpecialEdition/data/confusion_set/confusion.txt")

print(confusion_set[0])

confusion_dict = {}

for line in confusion_set:
	line = line.split(":")
	confusion_dict[line[0][0]] = line[-1]

print(confusion_dict["基"])

all_word_list = read_csv("/remote-home/xtzhang/CTC/CTC2021/SpecialEdition/scripts/sighan/all_word_dict.txt")

def wash_n(all_word_list):
	return [ re.sub("\n", "", i) for i in all_word_list]

all_word_list = wash_n(all_word_list)

print(all_word_list[:2])

trie = list2confusion_trie(all_word_list, confusion_dict)

t = trie.confusion["基"]
print(t)
print(len(trie.confusion.keys()))

test = trie.get_lexicon("法心社")

print(test)

t = trie.confusion["新"]
print(t)


time_ = time.time()
trie.assign("法新社")#记者法新社")
trie.mysearch(node=trie.root, pointer=0, is_mutated=False, main_key=0)
print(trie.result)
print(time.time() - time_)

def super_get(sentence):
	print("{")
	print("sentence: ", sentence)
	time_ = time.time()
	trie.assign(sentence)#记者法新社")
	#trie.my_search(node=trie.root, pointer=0, is_mutated=False, main_key=0)
	trie.my_get_lexion()
	print(trie.result)
	#print(trie.confusion_map)
	print("time:", time.time() - time_)
	print("}")
	return trie.result
print(trie.confusion["利"], trie.confusion["国"])

super_get("仲国")
super_get("美蝈人")
super_get("仲蝈")
super_get("巴蓦斯坦的大使馆")
super_get("法国驻巴蓦斯坦")

test_sentence = "法薪社记者报导,大使杰拉德,以及巴国官员在巴基斯坦西北部的托克哈姆边界关卡迎接十月九日与两名巴蓦斯坦同业一起遭塔莉班逮捕的裴哈。" 

patience = super_get("法薪社记者报导,大使杰拉德,以及巴国官员在巴基斯坦西北部的托克哈姆边界关卡迎接十月九日与两名巴蓦斯坦同业一起遭塔莉班逮捕的裴哈。")


print(sum(list(map(lambda x:len(x[2]), patience))))

#-----------------------------
import pickle
pair_dict = pickle.load(open("pair_dict.pickle", 'rb'))
single_dict = pickle.load(open("single_dict.pickle", 'rb'))

#print(pair_dict[('法', '法')])

from tqdm import tqdm
#tencent_embedding = read_csv("/remote-home/xtzhang/CTC/CTC2021/SpecialEdition/data/embed_corpus/Tencent_AILab_ChineseEmbedding.txt")
#word_dict = {}
#for line in tqdm(tencent_embedding[1:]):
#	word_dict[line.split()[0]] = 0
#path = "/remote-home/xtzhang/CTC/CTC2021/SpecialEdition/milestone/Chinese_from_dongxiexidian/dict.dat"
path = "./30wdict_utf8.txt"
dict_ = read_csv(path)
word_dict = {}
for line in tqdm(dict_):
	word_dict[re.sub("\W*", "", line)] = 0

print(list(word_dict.keys())[:10])

def get_combines(sentence, word):
	result = []
	for i in sentence:
		for j in word:
			result.append((i, j))
	return result
	
scores = {}

print(len(patience))

import math
for tri_set in patience:
	word = tri_set[-1]

	score = 0	
	
	if word not in word_dict:
		continue

	for c in word:
		for s in re.sub("\W*", "", test_sentence):
			try:
				p_xy = pair_dict[(c, s)] / 457010995 + pair_dict[(s, c)] / 457010995
			except:
				p_xy = 0
			try:
				p_x = single_dict[c] / 457010995
			except:
				p_x = 0
			try:
				p_y = single_dict[s] / 457010995
			except:
				p_y = 0
			#print(p_xy, p_x, p_y)
			score += p_xy* math.log( ( p_xy + 1e-10 ) / (p_x * p_y + 1e-10) )

	scores[tuple(tri_set)] = score

keys = list(scores.keys())

keys.sort(key= lambda x:scores[x], reverse=True)

for key in keys:
	print(scores[key], key)

print(len(scores.keys()))

