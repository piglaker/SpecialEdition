
import re

import tokenizers
import transformers
from transformers import (
    AutoTokenizer,
    AutoModel,
)


raw = "[2418  711  800 2110 2533 2523 1962 8024 2792  809 1398 4408 1398 2110 \
 6963 1599 3614 7309  800 7309 7579  511]"

raw2 = "1728.  711.  800. 2110. 4638. 2523. 1962. 8024. 2792.  809. 1398. 4408. \
 1398. 2110. 6963. 1599. 3614. 7309.  800. 7309. 7579.  511.]"

raw3 = "[1728  711  800 2110 2533 2523 1962 8024 2792  809 1398 4408 1398 2110 \
 6963 1599 3614 7309  800 7309 7579  511]"

tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")


raw = "[5445  684 5439 2360  738 2523 7410 3136 2360  511]"
raw2 = "[5445.  684. 5439. 2360.  738. 2523. 7410. 3136. 2360.  511.]"
raw3 = "[5445  684 5439 2360  738 2523 7410 3136  741  511]"

def show(raw):

    inputs = list(map(int, re.sub("\.", "", raw[1:-1]).split()))

    result = tokenizer.decode(inputs)

    print(result)


show(raw)

show(raw2)

show(raw3)
