#!/bin/bash

# for DATASET in cora citeseer wiki amazon-photo amazon-computers pubmed
# do
#     python -u execute.py --dataset $DATASET
# done

# for SEED in {0..2}
# do
#     DATASET=ogbn-arxiv_0.1_${SEED}_kd.npz
#     python -u execute.py --dataset $DATASET
# done

python -u execute.py --dataset cora-full



