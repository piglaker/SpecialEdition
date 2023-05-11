## Chinese Spell Check


训练纠错模型的代码对于ACL 2023 (Findings): [Investigating Glyph Phonetic Information for Chinese Spell Checking: What Works and What's Next](https://arxiv.org/abs/2212.04068)

论文中分析及Probe 指标见另一github仓库[ConfusionCluster](https://github.com/piglaker/ConfusionCluster)


1.Install all the requirements.  

use ./scripts/sighan/generate.py to generate data in ./data/rawdata/sighan

2.`bash run.sh` 


## Start-up

python >= 3.7  创建conda环境    
`conda create -n ctcSE python=3.7` 

then 安装必要包  
`conda activate ctcSE` \
`pip3 install -r requirements.txt` 

install nvcc 安装nvcc
略

apex  安装apex用于分布式训练  
`bash install_apex`  
or  
```
git clone https://github.com/NVIDIA/apex  
cd apex  
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" . 
```


install pytorch for your CUDA & GPU 安装gpu version的torch     
example:  
`conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 -c pytorch`  
or  
`pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 -f https://download.pytorch.org/whl/torch_stable.html`

maybe forget  安装datasets库
`pip install datasets==1.2.0`  

test env   测试环境是否正确
`sh test.sh"`

### Data

原始训练数据来自[Training Dataset](https://github.com/wdimmy/Automatic-Corpus-Generation)
处理后：分为raw和holy，  

下载并解压后分别放在如下路径：
原始版本:./data/rawdata/sighan/raw  
去重版本:./data/rawdata/sighan/holy  

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



