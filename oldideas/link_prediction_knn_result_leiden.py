
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

import leidenalg
import igraph as ig

def node2partition(partition):
    dic = {}
    for cl, nodes in enumerate(partition):
        for node in nodes:
            dic[node] = cl
    preds = []
    for i in range(max(dic.keys())+1):
        preds.append(dic[i])
    return preds

def leiden_cluster(adj, labels, random_state=None):
    from community import community_louvain
    import networkx as nx
    from sklearn.metrics import normalized_mutual_info_score as NMI, adjusted_mutual_info_score as AMI
    import igraph as ig

    graph = nx.from_scipy_sparse_matrix(adj)
    graph = ig.Graph.from_networkx(graph)
    partition = leidenalg.find_partition(graph, leidenalg.ModularityVertexPartition, seed=random_state)
    preds = node2partition(partition)

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

    seeds = np.arange(0, args.nexp, dtype=int) # seed = 0
    # seeds = [11]
    # seeds = [7]
    for seed in tqdm(seeds, total=len(seeds)):
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)
        torch.cuda.manual_seed(seed)

        setproctitle.setproctitle("LKR-{}-{}".format(dataset[:2], seed))

        adj, features, true_labels = load_assortative(dataset)
        n = adj.shape[0]
        m = adj.sum()
        raw_adj = adj.copy()


        data = np.load("link_adj_knn/{}/{}_{}.npz".format(args.model, args.dataset, seed))
        adj_data, adj_row, adj_col = data["data"], data["row"], data["col"]

        adj = sp.coo_matrix((adj_data, (adj_row, adj_col)), shape=(n, n))
        adj.eliminate_zeros()
        m2 = adj.sum()

        nmi_m, ami_m, ari_m, dbi_m = {}, {}, {}, {}
        edges = np.arange(m, 11*m, m, dtype=int)
        for sampling_edges in edges:
            
            if m2 > sampling_edges:
                sampling_rate =  sampling_edges / m2
                adj_s = sampling(adj, rate=sampling_rate, random_state=seed)
            else:
                adj_s = adj

            labels = true_labels
            preds = leiden_cluster(adj_s, labels, random_state=seed) 

            nmi = NMI(labels, preds)
            ami = AMI(labels, preds)
            ari = ARI(labels, preds)
            # dbi = -DBI(adj_s.toarray(), preds)
            # dbi = -DBI(sm_fea_s, preds)

            nmi_m[sampling_edges] = nmi
            ami_m[sampling_edges] = ami
            ari_m[sampling_edges] = ari
            # dbi_m[sampling_edges] = dbi

            os.makedirs("link_adj_knn/Cluster_leiden/{}".format(args.model), exist_ok=True)
            np.savez("link_adj_knn/Cluster_leiden/{}/lo_{}_preds_{}_{:.1f}.npz".format(args.model, args.dataset, seed, sampling_edges / m), preds=preds)



            
