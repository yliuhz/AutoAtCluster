
#!/bin/bash

for SEED in {0..9}
do

# https://github.com/zyzisastudyreallyhardguy/Graph-Group-Discrimination#:~:text=amco/data%22%20folder.-,For%20Amazon%20Photo,-python%20train_coauthor.py
python train_coauthor.py --n-classifier-epochs 2000 --n-hidden 512 --n-ggd-epochs 2000 --ggd-lr 0.0005 --proj_layers 1 --dataset_name 'photo' --seed $SEED

# https://github.com/zyzisastudyreallyhardguy/Graph-Group-Discrimination#:~:text=1%20%2D%2Ddataset_name%20%27photo%27-,For%20Amazon%20Computer,-%2Dn%2Dclassifier%2Depochs
python train_coauthor.py --n-classifier-epochs 3500 --n-hidden 1024 --n-ggd-epochs 1500 --ggd-lr 0.0001 --proj_layers 1 --dataset_name 'computer' --seed $SEED

done