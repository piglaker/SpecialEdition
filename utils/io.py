import json


def load_json(fp):
    f = open(fp, "r", encoding="utf-8")
    data = json.load(f)
    f.close()
    return data

def read_csv(path):
    result = []
    try:
        with open(path) as f:
            for line in f.readlines():
                #result.append(line.replace(" ", ""))
                result.append(line)
        return result
    except Exception as e:
        print(e)
        return 

def write_to(path, contents):
        f = open(path, "w", encoding='utf-8')
        f.write(contents)
        f.close()

def test():
    """
    
    """
    
    #Do something

    return


if __name__ == "__main__":
    test()
