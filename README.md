## Chinese Spell Check


1.Install all the requirements. \
2.Processe the fxxking data. \
3.`bash run.sh` 


use ./scripts/sighan/generate.py to generate data in ./data/rawdata/sighan


## Start-up

python >= 3.7 \
`conda create -n ctcSE python=3.7` 

apex \
`git clone https://github.com/NVIDIA/apex` \
`cd apex` \
`python setup.py install` 


install the fxxk pytorch for your CUDA & GPU \
example: \
`pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113` 

maybe forget \
`pip install datasets==1.2.0`

### Note:
    ./data
    ./models
    ./logs
    ./models
    ./scripts
    ./utils
    
core: \
    metric \
    load_model \
    load_dataset \
    args_process 

main: \
    out/err redirect 

lib: \
    hack transformers' trainer 



