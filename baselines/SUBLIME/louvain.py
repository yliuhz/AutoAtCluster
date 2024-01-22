
import numpy as np
import os
import scipy.sparse as sp
from community import community_louvain
import networkx as nx
from sklearn.metrics import normalized_mutual_info_score as NMI, adjusted_mutual_info_score as AMI, adjusted_rand_score as ARI

# from load import load_assortative
# from vis import plot_superadj

import setproctitle
import random

from data_loader import load_data
import argparse

def louvain_cluster(adj, labels, random_state=None):
    graph = nx.from_scipy_sparse_matrix(adj)
    partition = community_louvain.best_partition(graph, random_state=random_state)
    preds = list(partition.values())

    return preds

from tqdm import tqdm
import numpy as np
def sampling(adj, rate=0.5, random_state=None):
    n = adj.shape[0]
    adj = adj.toarray()
    
    ret = np.zeros((n,n))
    
    for i in range(n):
        row_idx = adj[i].nonzero()[0]
        arr = np.random.RandomState(seed=random_state).choice(row_idx, int(rate*row_idx.shape[0]))
        ret[i][arr] = 1
    
    return sp.coo_matrix(ret)

def make_parser():
    parser = argparse.ArgumentParser()
    # Experimental setting
    parser.add_argument('-dataset', type=str, default='pubmed')
    return parser


import setproctitle

if __name__ == "__main__":

    graph_learners = ["att", "mlp"]
    datasets = ["pubmed"]
    model = "SUBLIME"

    parser = make_parser()
    args = parser.parse_args()
    features, nfeats, labels, nclasses, train_mask, val_mask, test_mask, adj_original = load_data(args)

    num_edges = {
        "pubmed": adj_original.sum()
    }

    for graph_learner_idx, graph_learner in enumerate(graph_learners):
        for dataset in datasets:
            nseed=10
            # labels = true_labels[dataset]

            nmi_m, ami_m, ari_m = {}, {}, {}

            edges = np.arange(num_edges[dataset], 11*num_edges[dataset], num_edges[dataset], dtype=int)
            for m_idx, m in enumerate(edges):
                nmis, amis, aris = [], [], []

                for seed in range(nseed):
                    np.random.seed(seed)
                    random.seed(seed)

                    print(model, dataset, graph_learner_idx, m_idx, seed)
                    setproctitle.setproctitle("lolSB-{}-{}-{}".format(dataset, graph_learner_idx, m_idx, seed))

                    # try:
                    #     data = np.load("Cluster/{}/lo_{}_preds_{}_{}.npz".format(graph_learner, dataset, seed, m))
                    #     preds = data["preds"]

                    # except Exception as e:
                    data = np.load("outputs/learned_adj_{}_{}_{}.npz".format(graph_learner, dataset, seed))
                    adj_data, adj_row, adj_col = data["data"], data["row"], data["col"]

                    adj = sp.coo_matrix((adj_data, (adj_row, adj_col)), shape=(labels.shape[0], labels.shape[0]))
                    adj.eliminate_zeros()
                        
                    sampling_rate=m / adj.sum()
                    adj_s = sampling(adj, rate=sampling_rate)
                        
                    preds = louvain_cluster(adj_s, labels, random_state=seed)
                        
                    os.makedirs("Cluster_new/{}".format(graph_learner), exist_ok=True)
                    np.savez("Cluster_new/{}/lo_{}_preds_{}_{}.npz".format(graph_learner, dataset, seed, m), preds=preds)
                        
                        
    #                     pass
    # #                     print(e)
    #                 labels = true_labels[dataset]

    #                 nmi = NMI(labels, preds)
    #                 ami = AMI(labels, preds)
    #                 ari = ARI(labels, preds)

    #                 nmis.append(nmi)
    #                 amis.append(ami)
    #                 aris.append(ari)

    #             if len(nmis) > 0:
    #                 nmi_m[m] = np.mean(nmis)
    #                 ami_m[m] = np.mean(amis)
    #                 ari_m[m] = np.mean(aris)

    #         print(ari_m)