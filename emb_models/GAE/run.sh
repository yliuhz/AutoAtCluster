#!/bin/bash

for DATASET in cora citeseer wiki amazon-photo amazon-computers pubmed
do
    python -u gae.py --data $DATASET --model GAE
    python -u gae.py --data $DATASET --model VGAE
done