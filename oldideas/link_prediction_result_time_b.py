
import numpy as np
import pandas as pd
import random
from sklearn.metrics import normalized_mutual_info_score as NMI, adjusted_mutual_info_score as AMI, adjusted_rand_score as ARI, davies_bouldin_score as DBI
import time
from sklearn.preprocessing import normalize
import os

from utils import *
from vis import plot_superadj

from load import load_assortative


def louvain_cluster(adj, labels, random_state=None):
    from community import community_louvain
    import networkx as nx
    from sklearn.metrics import normalized_mutual_info_score as NMI, adjusted_mutual_info_score as AMI

    graph = nx.from_scipy_sparse_matrix(adj)
    partition = community_louvain.best_partition(graph, random_state=random_state)
    preds = list(partition.values())

    return preds


from tqdm import tqdm
def sampling(adj, rate=0.5, random_state=None):
    n = adj.shape[0]
    adj = adj.toarray()
    
    ret = np.zeros((n,n))
    
    for i in range(n):
        row_idx = adj[i].nonzero()[0]
        arr = np.random.RandomState(seed=random_state).choice(row_idx, int(rate*row_idx.shape[0]))
        ret[i][arr] = 1
    
    return sp.coo_matrix(ret)

import setproctitle


import argparse
def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cora', help='type of dataset.')
    parser.add_argument('--nexp', type=int, default=10, help="Number of repeated experiments")
    parser.add_argument("--save_model", action='store_true', help='Whether to store the link model')
    parser.add_argument("--model", type=str, default="GAE")
    parser.add_argument("--pos", type=float, default=0.01)
    parser.add_argument("--neg", type=float, default=0.9)
    parser.add_argument("--gnnlayers", type=int, default=2)
    

    ## Grid search
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--scaler", type=str, default="minmax")
    parser.add_argument("--mlp_layers", type=int, default=2)
    parser.add_argument("--epoch", type=int, default=100)


    return parser



if __name__ == "__main__":
    parser = make_parser()
    
    args = parser.parse_args()
    print(args)

    dataset = args.dataset
    print("dataset={}".format(dataset))

    nclasses = {
        "cora": 7,
        "citeseer": 6,
        "pubmed": 3,
    }

    xx = np.arange(1, 11, dtype=int).tolist()
    xx = [str(x) + "m" for x in xx]
    df_data_nmi = pd.DataFrame(columns=["models", "dataset", "params"]+xx)
    df_data_ami = pd.DataFrame(columns=["models", "dataset", "params"]+xx)
    df_data_ari = pd.DataFrame(columns=["models", "dataset", "params"]+xx)
    df_data_dbi = pd.DataFrame(columns=["models", "dataset", "params"]+xx)
    model = "Ours"

    adj, features, true_labels = load_assortative(dataset)
    n = adj.shape[0]
    m = adj.sum()
    raw_adj = adj.copy()

    seeds = np.arange(0, args.nexp, dtype=int) # seed = 0
    # seeds = [11]
    # seeds = [7]
    lo_time_m = []
    edges = np.arange(m, 11*m, m, dtype=int)
    edges = [1]
    for sampling_edges in edges:
        lo_times = []
        for seed in tqdm(seeds, total=len(seeds)):
            np.random.seed(seed)
            torch.manual_seed(seed)
            random.seed(seed)
            torch.cuda.manual_seed(seed)

            setproctitle.setproctitle("LKT-{}-{}".format(dataset[:2], seed))

            data = np.load("link_adj/{}/{}_{}.npz".format(args.model, args.dataset, seed))
            adj_data, adj_row, adj_col = data["data"], data["row"], data["col"]

            adj = sp.coo_matrix((adj_data, (adj_row, adj_col)), shape=(n, n))
            adj.eliminate_zeros()
            m2 = adj.sum()
            
            st = time.process_time()

            labels = true_labels
            preds = louvain_cluster(adj, labels, random_state=seed) 

            ed_lo = time.process_time()

            lo_times.append(ed_lo-st)

        lo_time_m.append(np.mean(lo_times))

    with open("Baseline_time.txt", "a+") as f:
        f.write("{}\n".format(args.dataset))
        f.write("lo_times: {}\n, {}\n".format(lo_times, lo_time_m))
        




            
