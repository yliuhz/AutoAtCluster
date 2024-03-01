#!/bin/bash

# python -u louvain.py
LOGDIR="logs"



# ## Cluster on unattributed graphs
# com-orkut
# DATAPATH=/data/yliumh/AutoAtClusterDatasets/snap/com-orkut.ungraph.txt
# for NTHREADS in 64 32 16 8 4 2 1
# do
# python -u louvainnk.py --mode "all" --unattr --datapath $DATAPATH --nseeds 3 --nthreads $NTHREADS 2>&1 | tee -a $LOGDIR/LouvainNK-formal-all-unattr.log
# python -u louvainnk.py --mode "queue" --unattr --datapath $DATAPATH --nseeds 3 --nthreads $NTHREADS 2>&1 | tee -a $LOGDIR/LouvainNK-formal-queue-unattr.log
# done
# # uk-2007-05
# DATAPATH=/data/yliumh/AutoAtClusterDatasets/networkit/uk-2007-05-edgelist.txt
# for NTHREADS in 64 32 16 8 4 2 1
# do
# python -u louvainnk.py --mode "all" --unattr --datapath $DATAPATH --nseeds 3 --nthreads $NTHREADS 2>&1 | tee -a $LOGDIR/LouvainNK-formal-all-unattr.log
# python -u louvainnk.py --mode "queue" --unattr --datapath $DATAPATH --nseeds 3 --nthreads $NTHREADS 2>&1 | tee -a $LOGDIR/LouvainNK-formal-queue-unattr.log
# done

# # networkit.ParallelLeiden
# # com-orkut
# DATAPATH=/data/yliumh/AutoAtClusterDatasets/snap/com-orkut.ungraph.txt
# for NTHREADS in 64 32 16 8 4 2 1
# do
# python -u leidennk.py --unattr --datapath $DATAPATH --nseeds 3 --nthreads $NTHREADS 2>&1 | tee -a $LOGDIR/LeidenNK-formal-unattr.log
# done
# # uk-2007-05
# DATAPATH=/data/yliumh/AutoAtClusterDatasets/networkit/uk-2007-05-edgelist.txt
# for NTHREADS in 64 32 16 8 4 2 1
# do
# python -u leidennk.py --unattr --datapath $DATAPATH --nseeds 3 --nthreads $NTHREADS 2>&1 | tee -a $LOGDIR/LeidenNK-formal-unattr.log
# done


# com-orkut
# DATAPATH=/data/yliumh/AutoAtClusterDatasets/snap/com-orkut.ungraph.txt
# for NTHREADS in 64 32 16 8 4 2 1
# do
# python -u louvainnk.py --mode "weighted" --unattr --datapath $DATAPATH --nseeds 3 --nthreads $NTHREADS 2>&1 | tee -a $LOGDIR/LouvainNK-formal-weighted-unattr.log #slow
# python -u louvainnk.py --mode "unweighted" --unattr --datapath $DATAPATH --nseeds 3 --nthreads $NTHREADS 2>&1 | tee -a $LOGDIR/LouvainNK-formal-unweighted-unattr.log #slow
# python -u louvainnk.py --mode "ps" --unattr --datapath $DATAPATH --nseeds 3 --nthreads $NTHREADS 2>&1 | tee -a $LOGDIR/LouvainNK-formal-ps-unattr.log #slow
# done
# uk-2007-05
# DATAPATH=/data/yliumh/AutoAtClusterDatasets/networkit/uk-2007-05-edgelist.txt
# for NTHREADS in 64 32 16 8 4 2 1
# do
# # python -u louvainnk.py --mode "weighted" --unattr --datapath $DATAPATH --nseeds 3 --nthreads $NTHREADS 2>&1 | tee -a $LOGDIR/LouvainNK-formal-weighted-unattr.log #slow
# python -u louvainnk.py --mode "unweighted" --unattr --datapath $DATAPATH --nseeds 3 --nthreads $NTHREADS 2>&1 | tee -a $LOGDIR/LouvainNK-formal-unweighted-unattr.log #slow
# python -u louvainnk.py --mode "ps" --unattr --datapath $DATAPATH --nseeds 3 --nthreads $NTHREADS 2>&1 | tee -a $LOGDIR/LouvainNK-formal-ps-unattr.log #slow
# done

# unattributed baselines
DATAROOT=/data/yliumh/AutoAtClusterDatasets/extractStructures
for DATANAME in cora citeseer wiki pubmed amazon-photo amazon-computers cora-full ogbn-arxiv
do
python -u louvainnk.py --mode "all" --unattr --datapath $DATAROOT/$DATANAME.edgelist --nseeds 3 --nthreads 32 2>&1 | tee -a $LOGDIR/LouvainNK-formal-all-unattr.log
python -u leidennk.py --unattr --datapath $DATAROOT/$DATANAME.edgelist --nseeds 3 --nthreads 32 2>&1 | tee -a $LOGDIR/LeidenNK-formal-unattr.log

# python -u louvainnk.py --mode "all" --unattr --datapath $DATAROOT/$DATANAME.edgelist --nseeds 3 --nthreads 1 2>&1 | tee -a $LOGDIR/LouvainNK-formal-all-unattr.log
# python -u leidennk.py --unattr --datapath $DATAROOT/$DATANAME.edgelist --nseeds 3 --nthreads 1 2>&1 | tee -a $LOGDIR/LeidenNK-formal-unattr.log
done

# Pubmed
# python -u louvainnk.py --mode "all"  --nthreads 64 2>&1 | tee -a $LOGDIR/LouvainNK-formal-all.log
# python -u louvainnk.py --mode "all"  --nthreads 64 --bootstrap 2>&1 | tee -a $LOGDIR/LouvainNK-formal-all.log


## Cluster on attributed 

# for NTHREADS in 32 1 64 16 8 4 2 64 
# do
# python -u louvainnk.py --mode "all"  --nthreads $NTHREADS 2>&1 | tee -a $LOGDIR/LouvainNK-formal-all.log
# python -u louvainnk.py --mode "queue" --nthreads $NTHREADS 2>&1 | tee -a $LOGDIR/LouvainNK-formal-queue.log
# python -u louvainnk.py --mode "weighted" --nthreads $NTHREADS 2>&1 | tee -a $LOGDIR/LouvainNK-formal-weighted.log
# python -u louvainnk.py --mode "unweighted" --nthreads $NTHREADS 2>&1 | tee -a $LOGDIR/LouvainNK-formal-unweighted.log
# python -u louvainnk.py --mode "ps" --nthreads $NTHREADS 2>&1 | tee -a $LOGDIR/LouvainNK-formal-ps.log
# python -u leidennk.py --nthreads $NTHREADS 2>&1 | tee -a $LOGDIR/LeidenNK-formal.log

# bootstrap
# python -u louvainnk.py --mode "all"  --nthreads $NTHREADS --bootstrap 2>&1 | tee -a $LOGDIR/LouvainNK-formal-all.log
# python -u louvainnk.py --mode "queue" --nthreads $NTHREADS --bootstrap 2>&1 | tee -a $LOGDIR/LouvainNK-formal-queue.log
# python -u louvainnk.py --mode "weighted" --nthreads $NTHREADS --bootstrap 2>&1 | tee -a $LOGDIR/LouvainNK-formal-weighted.log
# python -u louvainnk.py --mode "unweighted" --nthreads $NTHREADS --bootstrap 2>&1 | tee -a $LOGDIR/LouvainNK-formal-unweighted.log
# python -u louvainnk.py --mode "ps" --nthreads $NTHREADS --bootstrap 2>&1 | tee -a $LOGDIR/LouvainNK-formal-ps.log
# python -u leidennk.py --nthreads $NTHREADS --bootstrap 2>&1 | tee -a $LOGDIR/LeidenNK-formal.log
# done