

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
import sys
import torch
import torch.nn as nn

import sklearn.preprocessing as preprocess
import os
from ogb.nodeproppred import DglNodePropPredDataset


def hdbscan_cluster(adj, features, random_state=None, labels=None):
    from community import community_louvain
    import networkx as nx
    from sklearn.metrics import normalized_mutual_info_score as NMI, adjusted_mutual_info_score as AMI
    # from sklearn.cluster import OPTICS
    from hdbscan import HDBSCAN
    from sklearn.manifold import TSNE
    from sklearn.metrics import adjusted_rand_score as ARI
    from sklearn.decomposition import PCA

    print(features.shape)

    minPts = np.arange(2,64,1, dtype=int).tolist()
    max_ari = 0.0
    max_preds = None
    for minpt in minPts:
        # print(minpt, type(minpt))
        cluster = HDBSCAN(min_cluster_size=minpt)
        # aff = normalize(features.toarray())
        # aff = np.matmul(aff, np.transpose(aff))
        # preds = cluster.fit_predict(aff)
        # pca_features = PCA(n_components=15).fit_transform(features)
        # tsne_features = TSNE(n_components=2).fit_transform(features)
        preds = cluster.fit_predict(features)

        # os.makedirs("pics", exist_ok=True)
        # plt.figure()
        # # tsne_z = TSNE(n_components=2).fit_transform(features)
        # plt.scatter(tsne_features[:, 0], tsne_features[:, 1], c=preds)
        # plt.savefig("pics/tsne_{}.png".format(minpt))

        ari = ARI(labels, preds)

        if ari > max_ari:
            max_ari = ari
            max_preds = preds

        print(minpt, np.unique(preds).shape, ari)

    return max_preds

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

def load_data(dataset): # {'pubmed', 'citeseer', 'cora'}
    """Load data."""

    import pickle as pkl
    import networkx as nx
    import scipy.sparse as sp
    import torch
    from sklearn import preprocessing

    def parse_index_file(filename):
        index = []
        for line in open(filename):
            index.append(int(line.strip()))
        return index

    def sample_mask(idx, l):
        """Create mask."""
        mask = np.zeros(l)
        mask[idx] = 1
        return np.array(mask, dtype=bool)

    if dataset in ["cora", "citeseer", "pubmed"]:
        names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
        objects = []

        for i in range(len(names)):
            '''
            fix Pickle incompatibility of numpy arrays between Python 2 and 3
            https://stackoverflow.com/questions/11305790/pickle-incompatibility-of-numpy-arrays-between-python-2-and-3
            '''
            with open("/data/yliumh/AutoAtClusterDatasets/gcn/gcn/data/ind.{}.{}".format(dataset, names[i]), 'rb') as rf:
                u = pkl._Unpickler(rf)
                u.encoding = 'latin1'
                cur_data = u.load()
                objects.append(cur_data)
            # objects.append(
            #     pkl.load(open("data/ind.{}.{}".format(dataset, names[i]), 'rb')))
        x, y, tx, ty, allx, ally, graph = tuple(objects)
        test_idx_reorder = parse_index_file(
            "/data/yliumh/AutoAtClusterDatasets/gcn/gcn/data/ind.{}.test.index".format(dataset))
        test_idx_range = np.sort(test_idx_reorder)


        if dataset == 'citeseer':
            # Fix citeseer dataset (there are some isolated nodes in the graph)
            # Find isolated nodes, add them as zero-vecs into the right position
            test_idx_range_full = range(
                min(test_idx_reorder), max(test_idx_reorder) + 1)
            tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
            tx_extended[test_idx_range - min(test_idx_range), :] = tx
            tx = tx_extended
            ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
            ty_extended[test_idx_range - min(test_idx_range), :] = ty
            ty = ty_extended

        features = sp.vstack((allx, tx)).tolil()
        features[test_idx_reorder, :] = features[test_idx_range, :]
        features = torch.FloatTensor(np.array(features.todense()))
        adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
        
        labels = np.vstack((ally, ty))
        labels[test_idx_reorder, :] = labels[test_idx_range, :]
        
        idx_test = test_idx_range.tolist()
        idx_train = range(len(y))
        idx_val = range(len(y), len(y) + 500)

        train_mask = sample_mask(idx_train, labels.shape[0])
        val_mask = sample_mask(idx_val, labels.shape[0])
        test_mask = sample_mask(idx_test, labels.shape[0])

        y_train = np.zeros(labels.shape)
        y_val = np.zeros(labels.shape)
        y_test = np.zeros(labels.shape)
        y_train[train_mask, :] = labels[train_mask, :]
        y_val[val_mask, :] = labels[val_mask, :]
        y_test[test_mask, :] = labels[test_mask, :]

        adj = adj.toarray()
        labels = labels.argmax(1)
        # idx = labels.argsort(0)
        # adj = adj[idx, :][:, idx]
        # labels = labels[idx]
        # features = features[idx]

        adj = sp.coo_matrix(adj)
        features = sp.coo_matrix(features)

        return adj, features, labels
    elif dataset == "wiki":
        f = open('/data/yliumh/AutoAtClusterDatasets/AGE/data/graph.txt','r')
        adj, xind, yind = [], [], []
        for line in f.readlines():
            line = line.split()
            
            xind.append(int(line[0]))
            yind.append(int(line[1]))
            adj.append([int(line[0]), int(line[1])])
        f.close()
        ##logger.info(len(adj))

        f = open('/data/yliumh/AutoAtClusterDatasets/AGE/data/group.txt','r')
        label = []
        for line in f.readlines():
            line = line.split()
            label.append(int(line[1]))
        f.close()

        f = open('/data/yliumh/AutoAtClusterDatasets/AGE/data/tfidf.txt','r')
        fea_idx = []
        fea = []
        adj = np.array(adj)
        adj = np.vstack((adj, adj[:,[1,0]]))
        adj = np.unique(adj, axis=0)
        
        labelset = np.unique(label)
        labeldict = dict(zip(labelset, range(len(labelset))))
        label = np.array([labeldict[x] for x in label])
        adj = sp.coo_matrix((np.ones(len(adj)), (adj[:,0], adj[:,1])), shape=(len(label), len(label)))

        for line in f.readlines():
            line = line.split()
            fea_idx.append([int(line[0]), int(line[1])])
            fea.append(float(line[2]))
        f.close()

        fea_idx = np.array(fea_idx)
        features = sp.coo_matrix((fea, (fea_idx[:,0], fea_idx[:,1])), shape=(len(label), 4973)).toarray()
        scaler = preprocessing.MinMaxScaler()
        #features = preprocess.normalize(features, norm='l2')
        features = scaler.fit_transform(features)
        # features = torch.FloatTensor(features)
        features = sp.coo_matrix(features)

        return adj, features, label
    elif dataset in ["ogbn-arxiv", "ogbn-products"]:
        dataset = DglNodePropPredDataset(name="{}".format(dataset))
        g, labels = dataset[0]
        edge_indices = g.adj().indices()
        n, m = labels.shape[0], edge_indices[0].shape[0]
        adj = sp.coo_matrix((np.ones(m), (edge_indices[0].numpy(), edge_indices[1].numpy())), shape=(n,n))
        features = g.ndata["feat"]
        features = sp.coo_matrix(features)

        if labels.ndim > 1:
            if labels.shape[1] == 1:
                labels = labels.view(-1)
            else:
                labels = labels.argmax(1)
        labels = labels.numpy()
        return adj, features, labels
    elif dataset in ["amazon-photo", "amazon-computers", "cora-full"]:
        map2names = {
            "amazon-photo": "/data/yliumh/AutoAtClusterDatasets/gnn-benchmark/data/npz/amazon_electronics_photo.npz",
            "amazon-computers": "/data/yliumh/AutoAtClusterDatasets/gnn-benchmark/data/npz/amazon_electronics_computers.npz",
            "cora-full": "/data/yliumh/AutoAtClusterDatasets/gnn-benchmark/data/npz/cora_full.npz",
        }

        data = np.load(map2names[dataset])
        # logger.info(list(data.keys()))
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

        return adj, features, labels
    else:
        raise NotImplementedError()

def load_wiki():
    f = open('/data/yliumh/AutoAtClusterDatasets/AGE/data/graph.txt', 'r')
    adj, xind, yind = [], [], []
    for line in f.readlines():
        line = line.split()

        xind.append(int(line[0]))
        yind.append(int(line[1]))
        adj.append([int(line[0]), int(line[1])])
    f.close()
    ##logger.info(len(adj))

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

    features = sp.csr_matrix(features)

    if label.ndim > 1:
        if label.shape[1] == 1:
            label = label.view(-1)
        else:
            label = label.argmax(1)

    return adj, features, label


def load_emb_best_model(dataset, seed):
    emb_paths = {
        "cora": "/home/yliumh/github/AutoAtCluster/emb_models/GGD/manual_version/outputs/GGD_{}_emb_{}.npz",
        "citeseer": "/home/yliumh/github/AutoAtCluster/emb_models/GGD/manual_version/outputs/GGD_{}_emb_{}.npz",
        "wiki": "/home/yliumh/github/AutoAtCluster/emb_models/AGE/outputs/AGE_{}_emb_{}.npz",
        "pubmed": "/home/yliumh/github/AutoAtCluster/emb_models/AGE/outputs/AGE_{}_emb_{}.npz",
        "amazon-photo": "/home/yliumh/github/AutoAtCluster/emb_models/GGD/manual_version/outputs/GGD_{}_emb_{}.npz",
        "amazon-computers": "/home/yliumh/github/AutoAtCluster/emb_models/GGD/manual_version/outputs/GGD_{}_emb_{}.npz",
        "cora-full": "/home/yliumh/github/graph2gauss/outputs/G2G_{}_emb_{}.npz",
        "ogbn-arxiv": "/home/yliumh/github/AutoAtCluster/emb_models/GGD/manual_version/outputs/GGD_{}_emb1_{}.npz",
        "ogbn-products": "/home/yliumh/github/AutoAtCluster/emb_models/GGD/manual_version/outputs/GGD_{}_emb1_{}.npz"
    }

    emb_path = emb_paths[dataset].format(dataset, seed)
    data = np.load(emb_path)

    return data["emb"]


if __name__ == "__main__":
    datasets = ["cora", "citeseer", "wiki", "pubmed", "amazon-photo", "amazon-computers", "cora-full", "ogbn-arxiv"]
    # datasets = ["cora"]

    nclasses = {
        "cora": 7,
        "citeseer": 6,
        "pubmed": 3,
        "wiki": 19,
        "amazon-photo": 8,
        "amazon-computers": 10
    }
    
    seeds = np.arange(3, dtype=int)
    # seeds = np.arange(1, dtype=int)

    os.makedirs("outputs_emb", exist_ok=True)

    for dataset in datasets:
        for seed in seeds:
            print(dataset, seed)

            np.random.seed(seed)
            random.seed(seed)

            setproctitle.setproctitle("HD-{}-{}".format(dataset[:2], seed))

            adj, features, labels = load_data(dataset)
            emb = load_emb_best_model(dataset, seed)

            preds = hdbscan_cluster(adj, emb, random_state=seed, labels=labels)

            np.savez("outputs_emb/{}_{}.npz".format(dataset, seed), preds=preds, labels=labels)

            