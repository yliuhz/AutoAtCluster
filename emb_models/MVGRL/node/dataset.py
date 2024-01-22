# from dgl.data import CoraDataset, CitationGraphDataset
from dgl.data import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset
from utils import preprocess_features, normalize_adj
from sklearn.preprocessing import MinMaxScaler
from utils import compute_ppr
import scipy.sparse as sp
import networkx as nx
import numpy as np
import os

import sklearn.preprocessing as preprocess
import torch

import pickle as pkl
import sys


# def download(dataset):
#     if dataset == 'cora':
#         return CoraDataset()
#     elif dataset == 'citeseer' or 'pubmed':
#         return CitationGraphDataset(name=dataset)
#     else:
#         return None


def download(dataset):
    if dataset == 'cora':
        return CoraGraphDataset()
    elif dataset == 'citeseer':
        return CiteseerGraphDataset()
    elif dataset == 'pubmed':
        return PubmedGraphDataset()
    elif dataset == "new_chameleon":
        pass
    else:
        return None



def load(dataset):
    # datadir = os.path.join('data', dataset)

    if dataset == "wiki":
        adj, features, label = load_wiki()
        diff = compute_ppr(nx.from_scipy_sparse_array(adj), 0.2)
        adj = normalize_adj(adj + sp.eye(adj.shape[0])).todense()
        return adj, diff, features, label, 0, 0, 0
    elif dataset in ["cora", "citeseer", "pubmed"]:
        adj, feat, labels, idx_train, idx_val, idx_test = load_citation(dataset)

        g = nx.from_scipy_sparse_array(adj)

        diff = compute_ppr(g, 0.2)

    elif dataset in ["amazon-photo", "amazon-computers", "cora-full"]:
        map2names = {
            "amazon-photo": "/data/yliumh/AutoAtClusterDatasets/gnn-benchmark/data/npz/amazon_electronics_photo.npz",
            "amazon-computers": "/data/yliumh/AutoAtClusterDatasets/gnn-benchmark/data/npz/amazon_electronics_computers.npz",
            "cora-full": "/data/yliumh/AutoAtClusterDatasets/gnn-benchmark/data/npz/cora_full.npz",
        }

        data = np.load(map2names[dataset])
        # print(list(data.keys()))
        adj_data, adj_indices, adj_indptr, adj_shape = data["adj_data"], data["adj_indices"], data["adj_indptr"], data["adj_shape"]
        attr_data, attr_indices, attr_indptr, attr_shape = data["attr_data"], data["attr_indices"], data["attr_indptr"], data["attr_shape"]
        labels = data["labels"]

        adj = sp.csr_matrix((adj_data, adj_indices, adj_indptr), shape=adj_shape).tocoo()
        features = sp.csr_matrix((attr_data, attr_indices, attr_indptr), shape=attr_shape).toarray()

        g = nx.from_scipy_sparse_array(adj)
        diff = compute_ppr(g, 0.2)
        adj = normalize_adj(adj + sp.eye(adj.shape[0])).todense()

        if labels.ndim > 1:
            if labels.shape[1] == 1:
                labels = labels.view(-1)
            else:
                labels = labels.argmax(1)

        return adj, diff, features, labels, 0, 0, 0

    # if not os.path.exists(datadir):
    #     os.makedirs(datadir)
    #     ds = download(dataset)
    #     adj = nx.to_numpy_array(ds.graph)
    #     diff = compute_ppr(ds.graph, 0.2)
    #     feat = ds.features[:]
    #     labels = ds.labels[:]

    #     idx_train = np.argwhere(ds.train_mask == 1).reshape(-1)
    #     idx_val = np.argwhere(ds.val_mask == 1).reshape(-1)
    #     idx_test = np.argwhere(ds.test_mask == 1).reshape(-1)
        
    #     np.save(f'{datadir}/adj.npy', adj)
    #     np.save(f'{datadir}/diff.npy', diff)
    #     np.save(f'{datadir}/feat.npy', feat)
    #     np.save(f'{datadir}/labels.npy', labels)
    #     np.save(f'{datadir}/idx_train.npy', idx_train)
    #     np.save(f'{datadir}/idx_val.npy', idx_val)
    #     np.save(f'{datadir}/idx_test.npy', idx_test)
    # else:
    #     adj = np.load(f'{datadir}/adj.npy')
    #     diff = np.load(f'{datadir}/diff.npy')
    #     feat = np.load(f'{datadir}/feat.npy')
    #     labels = np.load(f'{datadir}/labels.npy')
    #     idx_train = np.load(f'{datadir}/idx_train.npy')
    #     idx_val = np.load(f'{datadir}/idx_val.npy')
    #     idx_test = np.load(f'{datadir}/idx_test.npy')


    if dataset == 'citeseer':
        feat = preprocess_features(feat)

        epsilons = [1e-5, 1e-4, 1e-3, 1e-2]
        avg_degree = np.sum(adj) / adj.shape[0]
        epsilon = epsilons[np.argmin([abs(avg_degree - np.argwhere(diff >= e).shape[0] / diff.shape[0])
                                      for e in epsilons])]

        diff[diff < epsilon] = 0.0
        scaler = MinMaxScaler()
        scaler.fit(diff)
        diff = scaler.transform(diff)

    adj = normalize_adj(adj + sp.eye(adj.shape[0])).todense()

    if labels.ndim > 1:
        if labels.shape[1] == 1:
            labels = labels.view(-1)
        else:
            labels = labels.argmax(1)

    return adj, diff, feat, labels, idx_train, idx_val, idx_test


def load_wiki():
    f = open('/data/yliumh/AutoAtClusterDatasets/AGE/data/graph.txt', 'r')
    adj, xind, yind = [], [], []
    for line in f.readlines():
        line = line.split()

        xind.append(int(line[0]))
        yind.append(int(line[1]))
        adj.append([int(line[0]), int(line[1])])
    f.close()
    ##print(len(adj))

    f = open('/data/yliumh/AutoAtClusterDatasets/AGE/data/group.txt', 'r')
    label = []
    for line in f.readlines():
        line = line.split()
        label.append(int(line[1]))
    f.close()

    f = open('/data/yliumh/AutoAtClusterDatasets/AGE/data/tfidf.txt', 'r')
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

    if label.ndim > 1:
        if label.shape[1] == 1:
            label = label.view(-1)
        else:
            label = label.argmax(1)

    return adj, features, label


def load_cora_full_im(rate=0.1):
    filename = "/data/liuyue/New/SBM/mySBM/data_im/cora-full_{:.1f}.npz".format(rate)
    data = np.load(filename)

    adj_raw, features_raw, labels_raw, _, _, _ = load("cora-full")

    adj_data, adj_row, adj_col, features_load, labels_load, mask = data["data"], data["row"], data["col"], data["features"], data["labels"], data["mask"]
    adj_load = sp.coo_matrix((adj_data, (adj_row, adj_col)), shape=(labels_load.shape[0], labels_load.shape[0]))

    adj_mask = adj_raw.toarray()[mask,:][:,mask]
    assert (adj_mask - adj_load).sum() < 1e-7
    features_mask = features_raw.toarray()[mask]
    assert (features_mask - features_load).sum() < 1e-7

    g = nx.from_scipy_sparse_array(adj_load)
    diff = compute_ppr(g, 0.2)
    adj_load = normalize_adj(adj_load + sp.eye(adj_load.shape[0])).todense()

    return adj_load, diff, features_load, labels_load, mask

def load_cora_full_diff_cls(nclass=10, seed=None):
    filename = "/data/liuyue/New/SBM/mySBM/data_diff_cls/cora-full_{}_{}.npz".format(nclass, seed)
    data = np.load(filename)

    # adj_raw, features_raw, labels_raw, _, _, _ = load("cora-full")

    adj_data, adj_row, adj_col, features_load, labels_load, mask = data["data"], data["row"], data["col"], data["features"], data["labels"], data["mask"]
    adj_load = sp.coo_matrix((adj_data, (adj_row, adj_col)), shape=(labels_load.shape[0], labels_load.shape[0]))

    # adj_mask = adj_raw.toarray()[mask,:][:,mask]
    # assert (adj_mask - adj_load).sum() < 1e-7
    # features_mask = features_raw.toarray()[mask]
    # assert (features_mask - features_load).sum() < 1e-7

    g = nx.from_scipy_sparse_array(adj_load)
    diff = compute_ppr(g, 0.2)
    adj_load = normalize_adj(adj_load + sp.eye(adj_load.shape[0])).todense()

    return adj_load, diff, features_load, labels_load, mask

def load_ogbn_arxiv_im(rate=0.1, seed=None):
    filename = "/data/liuyue/New/SBM/mySBM/data_im/ogbn-arxiv_{:.1f}_{:d}_l.npz".format(rate, seed)
    data = np.load(filename)

    # adj_raw, features_raw, labels_raw = load_assortative("cora-full")

    adj_data, adj_row, adj_col, features_load, labels_load, mask = data["data"], data["row"], data["col"], data["features"], data["labels"], data["mask"]
    adj_load = sp.coo_matrix((adj_data, (adj_row, adj_col)), shape=(labels_load.shape[0], labels_load.shape[0]))

    # adj_mask = adj_raw.toarray()[mask,:][:,mask]
    # assert (adj_mask - adj_load).sum() < 1e-7
    # features_mask = features_raw.toarray()[mask]
    # assert (features_mask - features_load).sum() < 1e-7

    g = nx.from_scipy_sparse_array(adj_load)
    diff = compute_ppr(g, 0.2)
    adj_load = normalize_adj(adj_load + sp.eye(adj_load.shape[0])).todense()

    return adj_load, diff, features_load, labels_load, mask



def load_citation(dataset_str):
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("/data/yliumh/AutoAtClusterDatasets/gcn/gcn/data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("/data/yliumh/AutoAtClusterDatasets/gcn/gcn/data/ind.{}.test.index".format(dataset_str))
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

    features = features.toarray()

    return adj, features, labels, idx_train, idx_val, idx_test


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

if __name__ == '__main__':
    load('cora')
