

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

import time
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')  # - %(name)s
logger = logging.getLogger(__name__)


def hca_cluster(adj, features, random_state=None, labels=None):
    from community import community_louvain
    import networkx as nx
    from sklearn.metrics import normalized_mutual_info_score as NMI, adjusted_mutual_info_score as AMI, adjusted_rand_score as ARI
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics.pairwise import euclidean_distances

    # max_ari = 0.0
    # max_preds = None
    # max_time = 0.0

    # dist = euclidean_distances(features, features)
    # max_thres = dist.max()


    # dist_thres = np.arange(30, max_thres+1, 1)
    # for thres in dist_thres:
    #     st = time.process_time()

    #     cluster = AgglomerativeClustering(n_clusters=None, distance_threshold=thres, linkage="average")
    #     preds = cluster.fit_predict(features)

    #     ed = time.process_time()

    #     ari = ARI(labels, preds)

    #     if ari > max_ari:
    #         max_ari = ari
    #         max_preds = preds
    #         max_time = ed-st

    #     nclass = np.unique(preds).shape[0]
    #     if nclass == 1:
    #         break
        
    #     print("{}/{}".format(thres, max_thres), ari, np.unique(preds).shape)

    thres = 48 # got by param search
    st = time.process_time()
    cluster = AgglomerativeClustering(n_clusters=None, distance_threshold=thres, linkage="average")
    preds = cluster.fit_predict(features)
    ed = time.process_time()
    max_preds = preds
    max_time = ed-st

    return max_preds, max_time

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


def load_emb_best_model(dataset, seed): # TSNE_embs
    emb_paths = {
        "cora": "/home/yliumh/github/AutoAtCluster/baselines/TSNE_embs/GGD/{}_tsne_{}.npz",
        "citeseer": "/home/yliumh/github/AutoAtCluster/baselines/TSNE_embs/GGD/{}_tsne_{}.npz",
        "wiki": "/home/yliumh/github/AutoAtCluster/baselines/TSNE_embs/AGE/{}_tsne_{}.npz",
        "pubmed": "/home/yliumh/github/AutoAtCluster/baselines/TSNE_embs/AGE/{}_tsne_{}.npz",
        "amazon-photo": "/home/yliumh/github/AutoAtCluster/baselines/TSNE_embs/GGD/{}_tsne_{}.npz",
        "amazon-computers": "/home/yliumh/github/AutoAtCluster/baselines/TSNE_embs/GGD/{}_tsne_{}.npz",
        "cora-full": "/home/yliumh/github/AutoAtCluster/baselines/TSNE_embs/G2G/{}_tsne_{}.npz",
        "ogbn-arxiv": "/home/yliumh/github/AutoAtCluster/baselines/TSNE_embs/GGD/{}_tsne_{}.npz",
        "ogbn-products": "/home/yliumh/github/AutoAtCluster/baselines/TSNE_embs/GGD/{}_tsne_{}.npz"
    }

    emb_path = emb_paths[dataset].format(dataset, seed)
    data = np.load(emb_path)

    return data["emb"]


if __name__ == "__main__":
    # datasets = ["cora", "citeseer", "wiki", "pubmed", "amazon-photo", "amazon-computers", "cora-full", "ogbn-arxiv"]
    datasets = ["ogbn-arxiv"]
    
    seeds = np.arange(3, dtype=int)

    os.makedirs("outputs_tsne", exist_ok=True)

    for dataset in datasets:
        times = []
        for seed in seeds:
            print(dataset, seed)

            if not os.path.exists("outputs_tsne/{}_{}.npz".format(dataset, seed)):

                np.random.seed(seed)
                random.seed(seed)

                setproctitle.setproctitle("HCA-{}-{}".format(dataset[:2], seed))

                adj, features, labels = load_data(dataset)
                emb = load_emb_best_model(dataset, seed)

                logger.info(f"Algorithm start: {dataset} {seed}")
                preds, dur = hca_cluster(adj, emb, random_state=seed, labels=labels)
                times.append(dur)
                np.savez("outputs_tsne/{}_{}.npz".format(dataset, seed), preds=preds, labels=labels)

            else:
                print(f"Skip: {dataset} {seed}")
            
        with open("time.txt", "a+") as f:
            f.write("HCA {} \n".format(dataset))
            for t in times:
                f.write("{:.3f} ".format(t))
            f.write("\n\n")
            