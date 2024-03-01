#!/bin/bash

for DATASET in cora citeseer wiki pubmed amazon-photo amazon-computers cora-full ogbn-arxiv
# for DATASET in amazon-photo amazon-computers cora-full ogbn-arxiv
do
    python -u IL.py $DATASET
done