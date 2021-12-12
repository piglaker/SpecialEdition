"""
PostProcess
    ###
    # Turn prediction from 'through' to human-read like valid.tgt
    #   
    ###

    in_ : 
        ../../tmp/tst-csc/generate_predictions.txt
        ../../data/rawdata/ctc2021/valid.src
        ../../data/rawdata/ctc2021/valid.tgt
        ../../data/rawdata/ctc2021/valid.through
    out_:
        ../../tmp/tst-csc/valid.pre


"""
import os
import sys


def read_csv(path):
    """
    for ctc2021 we syn all the func read_csv (copied form /data/DatasetLoadingHelper.py)
    """
    result = []

    with open(path) as f:
        for line in f.readlines():
            result.append(line)

    return result

def write_csv(content, path):
    """
    content shall be iterable like :list["sentence"]
    """
    with open(path, "w") as f:
        for sentence in content:
            f.write(sentence)
    
    return 

def run():

    valid_source_path = "../../data/rawdata/ctc2021/valid.src"
    valid_target_path = "../../data/rawdata/ctc2021/valid.tgt"
    valid_through_path = "../../data/rawdata/ctc2021/valid.through"

    prdt_through_path = "../../tmp/tst-csc/generated_predictions.txt"

    valid_source = read_csv(valid_source_path)
    valid_target = read_csv(valid_target_path)
    valid_through = read_csv(valid_through_path)

    prdt_through = read_csv(prdt_through_path)
    prdt_target = []    

    for i in range(len(valid_source)):
        source = valid_source[i]
        
        through = prdt_through[i].split()
        
        def seq(through):
            """
            through[i] maybe:
                "7 9 金 额"
                "0 1 中"
                "10 i 况"
                "0 i e"
                etc .
            return :
                (index, index, content) etc .
            """
            
            result = []
            bucket = []

            launch_count = 0

            for element in through:
                
                if element.isdigit():
                    if launch_count == 2:
                        result.append([bucket[0], bucket[1], "".join(bucket[2:])])
                        launch_count = 0
                        bucket = []
                    bucket.append(element)
                    launch_count += 1
                else:
                    if launch_count == 2:
                        bucket.append(element)
                    else:
                        bucket.append(element)
                        launch_count += 1
            
            return result



        copy = ""

        i = 0
        while i < len(through):
            start = through[i]
            end = through[i+1]
            if start.isdigit():
                if end.isdigit():
                    do()
                else:
                    copy += source[:int(start)] + source[int()]




    content = []
    for i in range(len(valid_source)):
        
        content.append("\n".join(valid_source, valid_target,  prdt_target, valid_through, prdt_through)) 


    return








if __name__ == "__main__":
    run()
