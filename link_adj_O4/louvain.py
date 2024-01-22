
import numpy as np
import os
import scipy.sparse as sp
from community import community_louvain
import networkx as nx
from sklearn.metrics import normalized_mutual_info_score as NMI, adjusted_mutual_info_score as AMI, adjusted_rand_score as ARI

from load import load_assortative
from vis import plot_superadj

import setproctitle
import random

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

if __name__ == "__main__":

    models = [
        "GAE",
        "VGAE",
        "ARGA",
        "ARVGA",
        "AGE",
        "DGI",
        "MVGRL",
        "GRACE",
        "GGD"
    ]

    datasets = [
        "cora",
        "citeseer",
        "wiki",
        "pubmed",
        "amazon-photo",
        "amazon-computers"
    ]

    true_labels = {}
    num_edges = {}
    for dataset in datasets:
        adj, _, label = load_assortative(dataset)
        true_labels[dataset] = label
        num_edges[dataset] = adj.sum()
        print("{}: {} {}".format(dataset, label.shape, num_edges[dataset]))

    for model_idx, model in enumerate(models):
        for dataset in datasets:
            nmis, amis, aris = [], [], []
            for seed in range(10):
                print(model, dataset, seed)
                setproctitle.setproctitle("lolO4-{}-{}-{}".format(model, dataset[:2], seed))

                np.random.seed(seed)
                random.seed(seed)

                if not os.path.exists("Cluster/{}/lo_{}_preds_{}.npz".format(model, dataset, seed)):
                    os.makedirs("Cluster/{}".format(model), exist_ok=True)

                    labels = true_labels[dataset]
                    m = num_edges[dataset]
                
                    data = np.load("{}/{}_{}.npz".format(model, dataset, seed))
                    adj_data, adj_row, adj_col = data["data"], data["row"], data["col"]

                    adj = sp.coo_matrix((adj_data, (adj_row, adj_col)), shape=(labels.shape[0], labels.shape[0]))
                    
        #             sampling_rate = m / adj.sum()
                    sampling_rate = 0.1
                    adj_s = sampling(adj, sampling_rate, random_state=seed)

                    plot_superadj(adj_s, K=100, sparse=True, labels=labels, dataset="link_s_{}".format(dataset), vline=True)

                    print(adj.sum(), adj_s.sum())

            #         preds = louvain_cluster(adj, labels, random_state=seed)
            #         nmi = NMI(labels, preds)
            #         ami = AMI(labels, preds)
            #         ari = ARI(labels, preds)
            #         print(nmi, ami, ari)

                    preds = louvain_cluster(adj_s, labels, random_state=seed)

                    np.savez("Cluster/{}/lo_{}_preds_{}.npz".format(model, dataset, seed), preds=preds, labels=labels)
                else:
                    print("Found. Pass.")
                    pass


            #     nmi = NMI(labels, preds)
            #     ami = AMI(labels, preds)
            #     ari = ARI(labels, preds)
                    
            #     nmis.append(nmi)
            #     amis.append(ami)
            #     aris.append(ari)
                
            # nmi_m = np.mean(nmis)
            # ami_m = np.mean(amis)
            # ari_m = np.mean(aris)
            
            # df_nmi["{}".format(dataset)][model_idx] = nmi_m
            # df_ami["{}".format(dataset)][model_idx] = ami_m
            # df_ari["{}".format(dataset)][model_idx] = ari_m


