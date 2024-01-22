

from community import community_louvain
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_networkx
import numpy as np
from sklearn.preprocessing import normalize
import torch
import argparse
import random
import setproctitle

import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh
import sys
import torch
import torch.nn as nn

import sklearn.preprocessing as preprocess
import os


def kmeans_cluster(adj, features, nclass, random_state=None):
    from community import community_louvain
    import networkx as nx
    from sklearn.metrics import normalized_mutual_info_score as NMI, adjusted_mutual_info_score as AMI
    from sklearn.cluster import KMeans

    cluster = KMeans(n_clusters=nclass, random_state=random_state)
    preds = cluster.fit_predict(features)

    return preds

def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cora', help='type of dataset.')
    parser.add_argument('--nexp', type=int, default=10, help="Number of repeated experiments")
    return parser

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


def load_cora_full_im(rate=0.1, seed=None):
    filename = "/data/liuyue/New/SBM/mySBM/data_im/cora-full_{:.1f}_{:d}.npz".format(rate, seed)
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

if __name__ == "__main__":
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

    nclasses = {
        "cora": 7,
        "citeseer": 6,
        "pubmed": 3,
        "wiki": 19,
        "amazon-photo": 8,
        "amazon-computers": 10
    }
    
    seeds = np.arange(3, dtype=int)

    os.makedirs("outputs_im", exist_ok=True)


    rates = np.arange(0.1, 1.0, 0.1)
    for dataset in datasets:
        for rate in rates:
            for seed in seeds:
                print(rate, seed)

                np.random.seed(seed)
                random.seed(seed)

                setproctitle.setproctitle("Km-{:.1f}-{}".format(rate, seed))

                adj, features, labels, mask= load_cora_full_im(rate, seed)
                emb = load_emb(models[0], dataset, seed, rate=rate)

                nclass = 10

                preds = kmeans_cluster(adj, emb, nclass, random_state=seed)

                np.savez("outputs_im/{}_{:.1f}_{}.npz".format(dataset, rate, seed), preds=preds, labels=labels)

            