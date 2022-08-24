import os
import re
import sys
import json

#upper import 
sys.path.append("../../") 
from utils import levenshtein
from utils.io import load_json, write_to



path = "../../data/rawdata/sighan/realise/data/"

import pickle

test13 = pickle.load(open(path+"test.sighan13.pkl", "rb"))

print(test13[0].keys())

print(test13[0]["src"], test13[0]["tgt"])


src = [ k["src"] for k in test13] 
tgt = [ k["tgt"] for k in test13]

write_to("../../data/rawdata/sighan/raw/test13.src", "\n".join(src))
write_to("../../data/rawdata/sighan/raw/test13.tgt", "\n".join(tgt))

