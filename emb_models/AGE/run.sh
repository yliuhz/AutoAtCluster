#!/bin/bash

# #Wiki
# python train.py --dataset wiki --gnnlayers 1 --upth_st 0.0011 --lowth_st 0.1 --upth_ed 0.001 --lowth_ed 0.5

#Pubmed
python train.py --dataset pubmed --gnnlayers 35 --upth_st 0.0013 --lowth_st 0.7 --upth_ed 0.001 --lowth_ed 0.8
