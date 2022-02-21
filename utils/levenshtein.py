
import os
import json

import Levenshtein

def levenshtein4seq(string, target, costs=(1, 1, 1), only_edits=False, only_dist=False):
    """

        piglaker modified version : 
            return edits
        iterative_levenshtein(s, t) -> ldist
        ldist is the Levenshtein distance between the strings
        s and t.
        For all i and j, dist[i,j] will contain the Levenshtein
        distance between the first i characters of s and the
        first j characters of t
        costs: a tuple or a list with three integers (d, i, s)
               where d defines the costs for a deletion
                     i defines the costs for an insertion and
                     s defines the costs for a substitution
        demo:
            >>> result, edits = levenshtein4seq([1,2,3], [0,13])
            >>> print(result, edits)
            >>> 3 [('substitution', 1, 0), ('substitution', 2, 13), ('delete', 3)]
    """
    rows = len(string) + 1
    cols = len(target) + 1
    deletes, inserts, substitutes = costs

    dist = [[0 for x in range(cols)] for x in range(rows)] # dist = np.zeros(shape=(rows, cols))

    edits = [ [[] for x in range(cols)] for x in range(rows)]

    # source prefixes can be transformed into empty strings
    # by deletions:
    for row in range(1, rows):
        dist[row][0] = row * deletes
    # target prefixes can be created from an empty source string
    # by inserting the characters
    for col in range(1, cols):
        dist[0][col] = col * inserts

    for col in range(1, cols):
        for row in range(1, rows):
            if string[row - 1] == target[col - 1]:
                cost = 0
            else:
                cost = substitutes
            dist[row][col] = min(dist[row - 1][col] + deletes,
                                 dist[row][col - 1] + inserts,
                                 dist[row - 1][col - 1] + cost)  # substitution

            # record edit
            min_distance = dist[row][col]

            if min_distance == dist[row - 1][col] + deletes:
                edit = ("delete", row-1, string[row - 1], "") 
                edits[row][col] = edits[row-1][col] + [edit]

            elif min_distance == dist[row][col - 1] + inserts:
                edit = ("insert", col-1, "", target[col-1])
                edits[row][col] = edits[row][col-1] + [edit]

            elif min_distance == dist[row - 1][col - 1] + cost:
                if string[row-1] != target[col-1]:
                    edit = ("substitution", row-1, string[row-1], target[col-1])
                    edits[row][col] =  edits[row-1][col-1] + [edit]
                else:
                    edits[row][col] = edits[row-1][col-1] 

            # (op:Str, id:int, content:Union[int, ...])
    if only_edits:
        return edits[row][col]
    elif only_dist:
        return dist[row][col]
    else:
        return dist[row][col], edits[row][col]

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
