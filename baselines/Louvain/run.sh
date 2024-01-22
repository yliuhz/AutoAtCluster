#!/bin/bash

# python -u louvain.py

echo "" > pid.txt
nohup python -u mylouvain.py --mode "all" 2>&1 | tee myLouvain-formal-all.log &
echo $! >> pid.txt
nohup python -u mylouvain.py --mode "unweighted" 2>&1 | tee myLouvain-formal-unweighted.log &
echo $! >> pid.txt
nohup python -u mylouvain.py --mode "weighted" 2>&1 | tee myLouvain-formal-weighted.log &
echo $! >> pid.txt
nohup python -u mylouvain.py --mode "queue" 2>&1 | tee myLouvain-formal-queue.log &
echo $! >> pid.txt
