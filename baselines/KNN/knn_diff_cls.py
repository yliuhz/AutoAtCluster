import dgl
import torch
import torch.nn as nn

import numpy as np
import random
import networkx as nx
import sys
import pickle as pkl
import sklearn.preprocessing as preprocess
import scipy.sparse as sp
import os
import matplotlib.pyplot as plt

import setproctitle


from utils import *
from vis import plot_superadj


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def load_data(dataset_str): # {'pubmed', 'citeseer', 'cora'}
    """Load data."""

    if dataset_str == 'wiki':
        adj, features, label = load_wiki()
        return adj, features, label, 0, 0, 0

    elif dataset_str in ["cora", "citeseer", "pubmed"]:

        names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
        objects = []
        for i in range(len(names)):
            with open("/data/liuyue/New/AGE/data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
                if sys.version_info > (3, 0):
                    objects.append(pkl.load(f, encoding='latin1'))
                else:
                    objects.append(pkl.load(f))

        x, y, tx, ty, allx, ally, graph = tuple(objects)
        test_idx_reorder = parse_index_file("/data/liuyue/New/AGE/data/ind.{}.test.index".format(dataset_str))
        test_idx_range = np.sort(test_idx_reorder)

        if dataset_str == 'citeseer':
            # Fix citeseer dataset (there are some isolated nodes in the graph)
            # Find isolated nodes, add them as zero-vecs into the right position
            test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
            tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
            tx_extended[test_idx_range-min(test_idx_range), :] = tx
            tx = tx_extended
            ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
            ty_extended[test_idx_range-min(test_idx_range), :] = ty
            ty = ty_extended

        features = sp.vstack((allx, tx)).tolil()
        features[test_idx_reorder, :] = features[test_idx_range, :]
        adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

        labels = np.vstack((ally, ty))
        labels[test_idx_reorder, :] = labels[test_idx_range, :]

        idx_test = test_idx_range.tolist()
        idx_train = range(len(y))
        idx_val = range(len(y), len(y)+500)


        if labels.ndim > 1:
            if labels.shape[1] == 1:
                labels = labels.view(-1)
            else:
                labels = labels.argmax(1)

    elif dataset in ["amazon-photo", "amazon-computers", "cora-full"]:
        map2names = {
            "amazon-photo": "/data/liuyue/New/SBM/mySBM/data/amazon_electronics_photo.npz",
            "amazon-computers": "/data/liuyue/New/SBM/mySBM/data/amazon_electronics_computers.npz",
            "cora-full": "/data/liuyue/New/SBM/mySBM/data/cora_full.npz",
        }

        data = np.load(map2names[dataset])
        # print(list(data.keys()))
        adj_data, adj_indices, adj_indptr, adj_shape = data["adj_data"], data["adj_indices"], data["adj_indptr"], data["adj_shape"]
        attr_data, attr_indices, attr_indptr, attr_shape = data["attr_data"], data["attr_indices"], data["attr_indptr"], data["attr_shape"]
        labels = data["labels"]

        adj = sp.csr_matrix((adj_data, adj_indices, adj_indptr), shape=adj_shape).tocoo()
        features = sp.csr_matrix((attr_data, attr_indices, attr_indptr), shape=attr_shape).tocoo()

        if labels.ndim > 1:
            if labels.shape[1] == 1:
                labels = labels.reshape(-1)
            else:
                labels = labels.argmax(1)

        return adj, features, labels, 0, 0, 0

    return adj, features, labels, idx_train, idx_val, idx_test


def load_wiki():
    f = open('/data/liuyue/New/AGE/data/graph.txt', 'r')
    adj, xind, yind = [], [], []
    for line in f.readlines():
        line = line.split()

        xind.append(int(line[0]))
        yind.append(int(line[1]))
        adj.append([int(line[0]), int(line[1])])
    f.close()
    ##print(len(adj))

    f = open('/data/liuyue/New/AGE/data/group.txt', 'r')
    label = []
    for line in f.readlines():
        line = line.split()
        label.append(int(line[1]))
    f.close()

    f = open('/data/liuyue/New/AGE/data/tfidf.txt', 'r')
    fea_idx = []
    fea = []
    adj = np.array(adj)
    adj = np.vstack((adj, adj[:, [1, 0]]))
    adj = np.unique(adj, axis=0)

    labelset = np.unique(label)
    labeldict = dict(zip(labelset, range(len(labelset))))
    label = np.array([labeldict[x] for x in label])
    adj = sp.csr_matrix((np.ones(len(adj)), (adj[:, 0], adj[:, 1])), shape=(len(label), len(label)))

    for line in f.readlines():
        line = line.split()
        fea_idx.append([int(line[0]), int(line[1])])
        fea.append(float(line[2]))
    f.close()

    fea_idx = np.array(fea_idx)
    features = sp.csr_matrix((fea, (fea_idx[:, 0], fea_idx[:, 1])), shape=(len(label), 4973)).toarray()
    scaler = preprocess.MinMaxScaler()
    # features = preprocess.normalize(features, norm='l2')
    features = scaler.fit_transform(features)
    # features = torch.FloatTensor(features)

    features = sp.csr_matrix(features)

    if label.ndim > 1:
        if label.shape[1] == 1:
            label = label.view(-1)
        else:
            label = label.argmax(1)

    return adj, features, label

def load_cora_full_diff_cls(nclass=10, seed=None):
    filename = "/data/liuyue/New/SBM/mySBM/data_diff_cls/cora-full_{}_{}.npz".format(nclass, seed)
    data = np.load(filename)

    adj_raw, features_raw, labels_raw, _, _, _ = load_data("cora-full")

    adj_data, adj_row, adj_col, features_load, labels_load, mask = data["data"], data["row"], data["col"], data["features"], data["labels"], data["mask"]
    adj_load = sp.coo_matrix((adj_data, (adj_row, adj_col)), shape=(labels_load.shape[0], labels_load.shape[0]))

    adj_mask = adj_raw.toarray()[mask,:][:,mask]
    assert (adj_mask - adj_load).sum() < 1e-7
    features_mask = features_raw.toarray()[mask]
    assert (features_mask - features_load).sum() < 1e-7

    return adj_load, features_load, labels_load, mask


def load_emb(model, dataset, seed, rate=None, nclass=None):
    emb_paths = {
        "GAE": "/data/liuyue/New/SBM/mySBM/emb_models/GAE/outputs", 
        "VGAE": "/data/liuyue/New/SBM/mySBM/emb_models/GAE/outputs",
        "ARGA": "/data/liuyue/New/SBM/mySBM/emb_models/ARGA/ARGA/arga/outputs",
        "ARVGA": "/data/liuyue/New/SBM/mySBM/emb_models/ARGA/ARGA/arga/outputs",
        "AGE": "/data/liuyue/New/SBM/mySBM/emb_models/AGE/outputs",
        "DGI": "/data/liuyue/New/SBM/mySBM/emb_models/DGI/outputs",
        "MVGRL": "/data/liuyue/New/SBM/mySBM/emb_models/MVGRL/outputs",
        "GRACE": "/data/liuyue/New/SBM/mySBM/emb_models/GRACE/outputs",
        "GGD": "/data/liuyue/New/SBM/mySBM/emb_models/GGD/manual_version/outputs"
    }
    assert model in emb_paths.keys()

    GRACE_datasets = {
        "cora": "Cora",
        "citeseer": "CiteSeer", 
        "wiki": "Wiki", 
        "pubmed": "PubMed",
        "amazon-photo": "amazon-photo",
        "amazon-computers": "amazon-computers"
    }
    if model == "GRACE":
        dataset_ = GRACE_datasets[dataset]
    else:
        dataset_ = dataset

    import os
    emb_path = emb_paths[model]

    if rate is not None:
        emb_path = os.path.join(emb_path, "{}_{}_emb_{:.1f}_{:d}.npz".format(model, dataset_, rate, seed))    
    elif nclass is not None:
        emb_path = os.path.join(emb_path, "{}_{}_emb_{:d}_{:d}.npz".format(model, dataset_, nclass, seed))
    else:
        emb_path = os.path.join(emb_path, "{}_{}_emb_{}.npz".format(model, dataset_, seed))

    data = np.load(emb_path)

    return data["emb"]

def knn_graph(embeddings, k, non_linearity, i):
    embeddings = torch.FloatTensor(embeddings)
    embeddings = F.normalize(embeddings, dim=1, p=2)
    similarities = cal_similarity_graph(embeddings)
    similarities = top_k(similarities, k + 1)
    similarities = apply_non_linearity(similarities, non_linearity, i)
    return similarities.numpy()

if __name__ == "__main__":

    # datasets = ["cora", "citeseer", "pubmed", "wiki"]
    # datasets = ["amazon-photo", "amazon-computers"]
    # datasets = ["wiki"]
    datasets = ["cora-full"]
    models = [
        # "GAE",
        # "VGAE",
        # "ARGA",
        # "ARVGA",
        # "AGE",
        # "DGI",
        "MVGRL",
        # "GRACE",
        # "GGD"
    ]

    seeds = np.arange(3, dtype=int) # seeds = {0,1,2}
    # seeds = [0]
    nclasses = np.arange(2, 12, 2, dtype=int)
    for model in models:
        for dataset in datasets:
            for nclass in nclasses:
                for seed in seeds:
                    print(model, dataset, nclass, seed)

                    np.random.seed(seed)
                    random.seed(seed)

                    setproctitle.setproctitle("KNNdc-{:.1f}-{}".format(nclass, seed))

                    # adj, features, labels, _, _, _ = load_data(dataset)
                    adj, features, labels, mask = load_cora_full_diff_cls(nclass, seed)
                    emb = load_emb(model, dataset, seed, nclass=nclass)
                    # emb = emb[mask]

                    n = adj.shape[0]
                    m = adj.sum()
                    # edges = np.arange(10*m, 11*m, m, dtype=int)
                    m2 = 10*m

                    k = np.ceil(m2 / n)
                    non_linear = "relu"

                    knn_adj = knn_graph(emb, k, non_linear, i=6)
                    knn_adj = sp.coo_matrix(knn_adj)
                    knn_adj.eliminate_zeros()

                    # plot_superadj(knn_adj, K=100, sparse=True, labels=labels, dataset="knn_{}".format(nclass), vline=True)

                    # knn_adj = knn_adj.toarray()

                    # plt.close()
                    # plt.figure()
                    # plt.scatter([x for x in range(n)], knn_adj.sum(1))
                    # plt.savefig("pics/knn_nbs.png")

                    # print(knn_adj[:20, :20])

                    # exit(0)

                    # print(adj.shape, knn_adj.shape)
                    # # exit(0)

                    os.makedirs("outputs_diff_cls", exist_ok=True)
                    np.savez("outputs_diff_cls/knn_adj_{:d}_{}.npz".format(nclass, seed), data=knn_adj.data, row=knn_adj.row, col=knn_adj.col)

