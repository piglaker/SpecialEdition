

#reference: https://github1s.com/LeeSureman/Flat-Lattice-Transformer/blob/HEAD/V0/utils_.py
import re
import collections

class TrieNode:
    def __init__(self):
        self.children = collections.defaultdict(TrieNode)
        self.is_w = False

class Trie:
    def __init__(self, confusion=None):
        self.root = TrieNode()
        #bidirect graph

        if confusion:
            graph = {i:[] for i in list(set([ j for i in confusion.values() for j in i ] + list(confusion.keys())))}


            for main_key in confusion.keys():
                line = re.sub("\W*", "", confusion[main_key])

                keys = [i for i in line]

                graph[main_key] += keys

                for key in keys:
                    graph[key].append(main_key)

            for key in graph.keys():
                graph[key] = list(set(graph[key]))

            self.confusion = graph
        self.sentence = None
        self.length = None
        self.result = None
        self.confusion_map = []
        self.debug =False
    
    def insert(self,w):

        current = self.root
        for c in w:
            current = current.children[c]

        current.is_w = True

    def search(self,w):
        current = self.root

        for c in w:
            current = current.children.get(c)

            if current is None:
                return False

        if current.is_w:
            return True
        else:
            return False

    def get_lexicon(self, sentence):
        result = []
        for i in range(len(sentence)):
            current = self.root
            for j in range(i, len(sentence)):
                current = current.children.get(sentence[j])
                print(j)
                if current is None:
                    break

                if current.is_w:
                    result.append([i,j,sentence[i:j+1]])

        return result

    def is_w(self, node):
        if node is None:
            return False
        else:
            return node.is_w

    def assign(self, sentence):
        self.sentence = re.sub("\W*", "", sentence)
        self.length = len(self.sentence)
        self.result = []

    def collect(self, main_key, pointer, pair):
        tmp = [main_key, pointer, self.sentence[main_key:pointer+1]]
        if pair:
            for p in pair:
                if self.debug:print(p.key, p.value, tmp)
                key = p.key - tmp[0]
                tmp[-1] = tmp[-1][:key] + p.value + tmp[-1][key+1:]
        self.result.append(tmp)

    def mysearch(self, node, pointer=0, is_mutated=0, main_key=0, pair=[]):
        """
        trie.assign(sentence)
        trie.mysearch(node=trie.root, pointer=1, is_mutated=False, main_key=0)
        """
        if self.debug:print("Gate", main_key, pointer)

        if not self.length:
            return 
        elif pointer >= self.length:
            return
        elif is_mutated >= 2:
            return

        if main_key == pointer:
            if self.sentence[pointer] not in self.confusion:
                return
            for query in self.confusion[self.sentence[main_key]]:
                branch = node.children.get(query)
                if branch:
                    if self.debug:print("Get:", pointer, query) 
                    self.mysearch(node=branch, pointer=pointer+1, is_mutated=is_mutated+1, main_key=main_key, pair=pair+[Pair(pointer, query)])

        current = node.children.get(self.sentence[pointer])

        if self.is_w(current) and pointer != main_key and is_mutated >= 1:
            if self.debug:print("Word")
            self.collect(main_key=main_key, pointer=pointer, pair=pair)

        elif current is not None:
            if self.debug:print("Deeper")
            self.mysearch(node=current, pointer=pointer+1, is_mutated=is_mutated, main_key=main_key, pair=pair)

        if is_mutated < 2 and ( pointer != main_key ) :
            if self.debug:print("Mutate", main_key, pointer, self.sentence[pointer])
            if self.sentence[pointer] not in self.confusion:
                if self.debug:print("Fail !")
                return
            for query in self.confusion[self.sentence[pointer]]:
                if self.debug:print("Mutate", main_key, pointer, self.sentence[pointer])
                branch = node.children.get(query)
                if branch:
                    if self.debug:print("Get:", pointer, query)
                    #tmp = self.sentence[main_key:pointer] + query
                    #if self.search( tmp ):
                    #    self.result.append([main_key, pointer, tmp])
                    if self.is_w(branch):
                        self.collect(main_key=main_key, pointer=pointer, pair=pair+[Pair(pointer, query)])

                    self.mysearch(node=branch, pointer=pointer+1, is_mutated=is_mutated+1, main_key=main_key, pair=pair+[Pair(pointer, query)])
            if self.debug:print("Fail !")

    def my_get_lexion(self, debug=False):
        self.debug = debug
        for i in range(self.length):
            if self.debug:print("Fate: ")
            self.mysearch(self.root, pointer=i, is_mutated=0, main_key=i)

class Pair:
    def __init__(self, k, v):
        self.key = k
        self.value = v


def list2confusion_trie(word_list, confusion_dict):
    word_trie = Trie(confusion_dict)
    for word in word_list:
        word_trie.insert(word)

    return word_trie

def list2trie(word_list):
    """
    """
    word_trie = Trie()
    for word in word_list:
        word_trie.insert(word)

    return word_trie

def build_trie(pretrain_vocab_path):
    """
    """
    with open(pretrain_vocab_path, "r", encoding="utf-8") as fr:
        lines = fr.readlines()

    word_list = []
    for line in lines:
        line_item = line.strip().split(' ')
        word = line_item[0]
        word_list.append(word)

    word_trie = list2trie(word_list)

    return word_trie
