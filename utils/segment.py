from transformers import AutoConfig, AutoModel, AutoTokenizer


def tokenize(data, vocab_file_path="./vocab.txt"):
    import tokenization

    #tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file_path, do_lower_case=True)

    model_name_or_path = "hfl/chinese-roberta-wwm-ext"
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path,)

    result = []

    for line in data:
        line = line.strip()
        #tmp = ""
        
        #line = tokenization.convert_to_unicode(line)
        #if not line:
        #    result.append(tmp)
        #    continue

        #tokens = tokenizer.tokenize(line)

        tokens = tokenizer.tokenize(line)

        result.append(' '.join(tokens))

    return result

def demo():

    res = tokenize(["1947年 5月的孟良崮战役，九纵沿雕窝、东540高地向孟良崮珠峰突击，沿途伤亡很大。"], "./vocab.txt")

    print(res)

    return 


if __name__ == "__main__":
    demo()