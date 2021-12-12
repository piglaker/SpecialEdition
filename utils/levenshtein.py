
import os
import json

import Levenshtein


def tokenize(data, vocab_file_path="./vocab.txt"):
    import tokenization

    tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file_path, do_lower_case=True)

    result = []

    for line in data:
        line = line.strip()
        tmp = ""
        line = tokenization.convert_to_unicode(line)
        if not line:
            result.append(tmp)
            continue

        tokens = tokenizer.tokenize(line)

        result.append(' '.join(tokens))

    return result

def convert_from_sentpair_through(f_src, f_tgt, f_sid):

    import Levenshtein
    import unicodedata

    #f_sid = [sentence["original_text"] for sentence in f_sid]

    #print(len(f_src), len(f_tgt), len(f_sid))

    predict_result = []

    for i in range(len(f_src)):

        tmp = ""

        src_line = unicodedata.normalize('NFKC', f_src[i].strip().replace(' ', '').replace('“', '\"').replace('”', '\"'))

        tgt_line = unicodedata.normalize('NFKC', f_tgt[i].strip().replace(' ', '').replace('“', '\"').replace('”', '\"'))

        sid = f_sid[i].strip().split('\t')[0]

        edits = Levenshtein.opcodes(src_line, tgt_line)

        result = []

        for edit in edits:
            if ("。" or "“" or "," or "”" or "《" or "》") in src_line[edit[3]:edit[4]]: # rm 。
                continue
            if edit[0] == "insert":
                if "," not in tgt_line[edit[3]:edit[4]]:
                    result.append((str(edit[1]), "i", tgt_line[edit[3]:edit[4]]))
            elif edit[0] == "replace":
                if "," not in src_line[edit[1]:edit[2]] and "," not in tgt_line[edit[3]:edit[4]]:
                    result.append((str(edit[1]), str(edit[2]),  tgt_line[edit[3]:edit[4]]))
            elif edit[0] == "delete":
                if "," not in src_line[edit[1]:edit[2]]:
                    result.append((str(edit[1]), str(edit[2]), "d"))

        out_line = ""
        for res in result:
            out_line +=  ' '.join(res) + ' '
        if out_line:
            tmp += out_line.strip() 
        else:
            tmp += ""

        predict_result.append(tmp)

    return predict_result



def convert_from_sentpair_to(f_src, f_tgt):

    import Levenshtein
    import unicodedata

    predict_result = []

    for i in range(len(f_src)):

        tmp = ""

        src_line = unicodedata.normalize('NFKC', f_src[i].strip().replace(' ', '').replace('“', '\"').replace('”', '\"'))

        tgt_line = unicodedata.normalize('NFKC', f_tgt[i].strip().replace(' ', '').replace('“', '\"').replace('”', '\"'))

        edits = Levenshtein.opcodes(src_line, tgt_line)

        result = []

        for edit in edits:
            if ("。" or "“" or "," or "”" or "《" or "》") in src_line[edit[3]:edit[4]]: # rm 。
                continue
            if edit[0] == "insert":
                if "," not in tgt_line[edit[3]:edit[4]]:
                    result.append((str(edit[1]), "缺失", "", tgt_line[edit[3]:edit[4]]))
            elif edit[0] == "replace":
                if "," not in src_line[edit[1]:edit[2]] and "," not in tgt_line[edit[3]:edit[4]]:
                    result.append((str(edit[1]), "别字", src_line[edit[1]:edit[2]], tgt_line[edit[3]:edit[4]]))
            elif edit[0] == "delete":
                if "," not in src_line[edit[1]:edit[2]]:
                    result.append((str(edit[1]), "冗余", src_line[edit[1]:edit[2]], ""))

        out_line = ""
        for res in result:
            out_line +=  ', '.join(res) + ', '
        if out_line:
            tmp += out_line.strip() 
        else:
            tmp += '-1'

        predict_result.append(tmp)

    return predict_result



def demo():
    f_src = ["优点：反映科目之间的对应关系，便于了解经济业务概况，辩于检查和分析经问济业务；"]
    f_tgt = ["优点：反映科目之间的对应关系，便于了解经济业务概况，便于检查和分析经济业务；"]
    
    print("source: ", f_src)
    print("target: ", f_tgt)

    res = convert_from_sentpair_to(f_src, f_tgt)

    print(res)

    return


if __name__ == "__main__":
    demo()
