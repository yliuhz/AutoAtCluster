#!/bin/bash

# Train 1 epoch only as in paper
# for SEED in {0..9}
# do

# python -u train_arxiv_ready.py --dataset_name 'ogbn-arxiv' --dataset=ogbn-arxiv --ggd-lr 0.0001 --n-hidden 256 --n-layers 3 --proj_layers 1 --gnn_encoder 'gcn' --n-ggd-epochs 1 --seed $SEED

# done

# Train more epochs
for SEED in {0..9}
do

python -u train_arxiv_ready.py --dataset_name 'ogbn-arxiv' --dataset=ogbn-arxiv --ggd-lr 0.0001 --n-hidden 256 --n-layers 3 --proj_layers 1 --gnn_encoder 'gcn' --n-ggd-epochs 1500 --seed $SEED

done