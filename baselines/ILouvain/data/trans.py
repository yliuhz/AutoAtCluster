import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh
import sys
import torch
import torch.nn as nn
import os

import sklearn.preprocessing as preprocess

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
            with open("{}/ind.{}.{}".format(dataset_str, dataset_str, names[i]), 'rb') as f:
                if sys.version_info > (3, 0):
                    objects.append(pkl.load(f, encoding='latin1'))
                else:
                    objects.append(pkl.load(f))

        x, y, tx, ty, allx, ally, graph = tuple(objects)
        test_idx_reorder = parse_index_file("{}/ind.{}.test.index".format(dataset_str, dataset_str))
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

    elif dataset_str in ["amazon-photo", "amazon-computers"]:
        map2names = {
            "amazon-photo": "/data/liuyue/New/SBM/mySBM/data/amazon_electronics_photo.npz",
            "amazon-computers": "/data/liuyue/New/SBM/mySBM/data/amazon_electronics_computers.npz",
        }

        data = np.load(map2names[dataset_str])
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
    f = open('wiki/graph.txt', 'r')
    adj, xind, yind = [], [], []
    for line in f.readlines():
        line = line.split()

        xind.append(int(line[0]))
        yind.append(int(line[1]))
        adj.append([int(line[0]), int(line[1])])
    f.close()
    ##print(len(adj))

    f = open('wiki/group.txt', 'r')
    label = []
    for line in f.readlines():
        line = line.split()
        label.append(int(line[1]))
    f.close()

    f = open('wiki/tfidf.txt', 'r')
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
    # scaler = preprocess.MinMaxScaler()
    # # features = preprocess.normalize(features, norm='l2')
    # features = scaler.fit_transform(features)
    # features = torch.FloatTensor(features)

    features = sp.csr_matrix(features)

    if label.ndim > 1:
        if label.shape[1] == 1:
            label = label.view(-1)
        else:
            label = label.argmax(1)

    return adj, features, label


def load_emb(model, dataset, seed):
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
    emb_path = os.path.join(emb_path, "{}_{}_emb_{}.npz".format(model, dataset_, seed))

    data = np.load(emb_path)

    return data["emb"]


if __name__ == "__main__":

    datasets = ["cora", "citeseer", "pubmed", "wiki", "amazon-photo", "amazon-computers"]

    for dataset in datasets:


        adj, features, labels, _, _, _ = load_data(dataset)

        n, d = features.shape

        os.makedirs("{}".format(dataset), exist_ok=True)

        edges = adj.nonzero()
        with open("{}/{}.edgeList".format(dataset, dataset), "w") as f:
            for u, v in zip(*edges):
                f.write("{} {}\n".format(u, v))

        features = features.toarray()
        with open("{}/{}.attributes".format(dataset, dataset), "w") as f:
            for i in range(n):
                f.write("{} ".format(i))
                for idx, a in enumerate(features[i]):
                    f.write("{}".format(a))
                    if idx < d-1:
                        f.write(",")
                f.write("\n")

        for seed in range(10):
            print(dataset, seed)
            
            emb = load_emb("GRACE", dataset, seed)
            n, d = emb.shape
            with open("{}/{}_{}.emb".format(dataset, dataset, seed), "w") as f:
                for i in range(n):
                    f.write("{} ".format(i))
                    for idx, a in enumerate(emb[i]):
                        f.write("{}".format(a))
                        if idx < d-1:
                            f.write(",")
                    f.write("\n")

    
