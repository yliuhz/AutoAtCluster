

Official repository of "Attributed Community Detection without Cluster Cardinality"

Yue Liu, Zhongying Ru, Xiaofang Zhou


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

## 