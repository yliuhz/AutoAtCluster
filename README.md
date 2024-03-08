

# Scalable Attributed Node Clustering Without Cluster Number

[Yue Liu](https://yliuhz.github.io/), [Zhongying Ru](https://www.linkedin.com/in/%E9%92%9F%E8%8E%B9-%E8%8C%B9-8b4732187/?locale=cs_CZ), Xiaofang Zhou


## Installation

On a Linux machine with CUDA12.1 installed, run the following commands

```bash
git clone --recurse-submodules -b release https://github.com/yliuhz/AutoAtCluster.git && cd AutoAtCluster 
conda create -n autocls python=3.10.13
conda activate autocls
python -m pip install -r requirements.txt
git clone https://github.com/yliuhz/DeepRobust.git && cd DeepRobust && python setup.py install && cd -
cd networkit && python setup.py build_ext && python -m pip install -e . && cd -
```

## Get Started

Put your data into a directory, and then change the value of `DATAROOT` variable in `runNK.sh`. 

```bash
bash runNK.sh
```