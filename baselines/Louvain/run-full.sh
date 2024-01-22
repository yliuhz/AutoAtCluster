#!/bin/bash

# echo "DEBUG.py"
# python -u myLouvain-debug.py

# echo "FULL.py"
# python -u mylouvain-full.py


nohup python -u mylouvain-full.py --mode "all" 2>&1 | tee myLouvainFull-formal-all.log &
echo $! >> pid.txt
nohup python -u mylouvain-full.py --mode "unweighted" 2>&1 | tee myLouvainFull-formal-unweighted.log &
echo $! >> pid.txt
nohup python -u mylouvain-full.py --mode "weighted" 2>&1 | tee myLouvainFull-formal-weighted.log &
echo $! >> pid.txt
nohup python -u mylouvain-full.py --mode "queue" 2>&1 | tee myLouvainFull-formal-queue.log &
echo $! >> pid.txt