#!/bin/bash

for DATASET in cora citeseer wiki amazon-photo amazon-computers pubmed
do
    python node/train.py --dataset $DATASET --nexp 10
done