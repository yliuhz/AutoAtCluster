#!/bin/bash



# Cluster on attributed embeddings
python -u louvainnk-full.py --mode "all" 2>&1 | tee LouvainNKFull-formal-all.log
python -u louvainnk-full.py --mode "unweighted" 2>&1 | tee LouvainNKFull-formal-unweighted.log
python -u louvainnk-full.py --mode "weighted" 2>&1 | tee LouvainNKFull-formal-weighted.log
python -u louvainnk-full.py --mode "queue" 2>&1 | tee LouvainNKFull-formal-queue.log