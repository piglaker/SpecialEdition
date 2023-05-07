## Chinese Spell Check


训练纠错模型的代码对于ACL 2023 (Findings): [Investigating Glyph Phonetic Information for Chinese Spell Checking: What Works and What's Next](https://arxiv.org/abs/2212.04068)

论文中分析及Probe 指标见另一github仓库[ConfusionCluster](https://github.com/piglaker/ConfusionCluster)


1.Install all the requirements.  

use ./scripts/sighan/generate.py to generate data in ./data/rawdata/sighan

2.`bash run.sh` 


## Start-up

python >= 3.7 \
`conda create -n ctcSE python=3.7` 

then \
`conda activate ctcSE` \
`pip3 install -r requirements.txt` 


apex \
`bash install_apex` \
or \
`git clone https://github.com/NVIDIA/apex` \
`cd apex` \
`pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" .` 


install pytorch for your CUDA & GPU \
example: \
`conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 -c pytorch`  
or \
`pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 -f https://download.pytorch.org/whl/torch_stable.html`

maybe forget  
`pip install datasets==1.2.0`  

test env   
`sh test.sh"`


### Note:
dir:  
- ./data  
- ./models   
- ./logs   
- ./models   
- ./scripts   
- ./utils  
    
core: 
- metric  
- load_model  
- load_dataset 
- args_process 

main:  
    out/err redirect 

lib:  
    hack transformers' trainer 



