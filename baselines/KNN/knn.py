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

import setproctitle


from utils import *
from vis import plot_superadj
from ogb.nodeproppred import DglNodePropPredDataset
import time
from datetime import datetime

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')  # - %(name)s
logger = logging.getLogger(__name__)


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
    f = open('/data/liuyue/New/AGE/data/graph.txt', 'r')
    adj, xind, yind = [], [], []
    for line in f.readlines():
        line = line.split()

        xind.append(int(line[0]))
        yind.append(int(line[1]))
        adj.append([int(line[0]), int(line[1])])
    f.close()
    ##logger.info(len(adj))

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
        "GGD": "/data/liuyue/New/SBM/mySBM/emb_models/GGD/manual_version/outputs",
        "GGD_product": "/data/liuyue/New/SBM/mySBM/emb_models/GGD/GGD_ogbn_product_1epoch/outputs",
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
    emb_path = os.path.join(emb_path, "{}_{}_emb_{}.npz".format(model.split("_")[0], dataset_, seed))

    data = np.load(emb_path)

    return data["emb"]

"""
2023/12/29: Load embedding for each dataset, choosing the best model
"""
def load_emb_best_model(dataset, seed):
    emb_paths = {
        "cora": "/home/yliumh/github/AutoAtCluster/emb_models/GGD/manual_version/outputs/GGD_{}_emb_{}.npz",
        "citeseer": "/home/yliumh/github/AutoAtCluster/emb_models/GGD/manual_version/outputs/GGD_{}_emb_{}.npz",
        "wiki": "/home/yliumh/github/AutoAtCluster/emb_models/AGE/outputs/AGE_{}_emb_{}.npz",
        "pubmed": "/home/yliumh/github/AutoAtCluster/emb_models/GGD/manual_version/outputs/GGD_{}_emb_{}.npz",
        "amazon-photo": "/home/yliumh/github/AutoAtCluster/emb_models/GGD/manual_version/outputs/GGD_{}_emb_{}.npz",
        "amazon-computers": "/home/yliumh/github/AutoAtCluster/emb_models/GGD/manual_version/outputs/GGD_{}_emb_{}.npz",
        "cora-full": "/home/yliumh/github/graph2gauss/outputs/G2G_{}_emb_{}.npz",
        "ogbn-arxiv": "/home/yliumh/github/AutoAtCluster/emb_models/GGD/manual_version/outputs/GGD_{}_emb1_{}.npz"
    }

    emb_path = emb_paths[dataset].format(dataset, seed)
    data = np.load(emb_path)

    return data["emb"]

def knn_graph(embeddings, k, non_linearity, i):
    embeddings = torch.FloatTensor(embeddings)
    embeddings = F.normalize(embeddings, dim=1, p=2)
    similarities = cal_similarity_graph(embeddings)
    similarities = top_k(similarities, k + 1)
    similarities = apply_non_linearity(similarities, non_linearity, i)
    return similarities.numpy()


def batch_knn_graph(embeddings, k, non_linearity, i, batch_size=1024):
    embeddings = preprocess.normalize(embeddings)
    n, d = embeddings.shape
    data, row, col = [], [], []
    st = 0
    k = int(k)
    while st < n:
        ed = min(n, st+batch_size)
        sub_emb = embeddings[st:ed]
        sub_sim = np.matmul(sub_emb, np.transpose(embeddings))
        indices = np.argpartition(-sub_sim, kth=k, axis=-1)[:, :k].reshape(-1)

        row += (np.repeat(np.arange(st,ed,1, dtype=int), k).tolist())
        col += (indices.tolist())

        sub_row = np.repeat(np.arange(ed-st, dtype=int), k).tolist()
        sub_col = indices.tolist()
        assert len(sub_row) == len(sub_col)
        data += (sub_sim[sub_row, sub_col].tolist())
        
        st += batch_size

        logger.info(f"{st:d}/{n:d}")

    # logger.info(data[:5])
    # logger.info(row[:5])
    # logger.info(col[:5])
    return sp.coo_matrix((data, (row, col)), shape=(n,n))

if __name__ == "__main__":

    datasets = [
        "cora",
        "citeseer",
        "wiki",
        "pubmed",
        "amazon-photo",
        "amazon-computers",
        "cora-full",
        "ogbn-arxiv"
    ]

    seeds = np.arange(3, dtype=int)
    for dataset in datasets:
        for seed in seeds:
            logger.info(f"{dataset}, {seed}")

            np.random.seed(seed)
            random.seed(seed)

            setproctitle.setproctitle("KNN-{}-{}".format(dataset, seed))

            adj, features, labels = load_data(dataset)
            emb = load_emb_best_model(dataset, seed)

            n = adj.shape[0]
            m = adj.sum()
            edges = np.arange(m, 10*m, m, dtype=int)
            edges = np.concatenate([edges, np.arange(10*m, 101*m, 10*m, dtype=int)])

            for m2 in edges:
                if os.path.exists("outputs/knn_adj_{}_{}_{:.0f}.npz".format(dataset, seed, m2/m)):
                    logger.info("Skip")
                    continue

                k = np.ceil(m2 / n)
                non_linear = "relu"

                # knn_adj = knn_graph(emb, k, non_linear, i=6)
                alg_st = time.time()
                knn_adj = batch_knn_graph(emb, k, non_linear, i=6, batch_size=1024)
                knn_adj = sp.coo_matrix(knn_adj)
                knn_adj.eliminate_zeros()
                alg_end = time.time()

                time_now = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
                with open("time.txt", "a+") as f:
                    f.write(f"{dataset}\t{seed}\t{m2/m:.0f}\t{alg_end-alg_st}\t{time_now}\n")

                # plot_superadj(knn_adj, K=100, sparse=True, labels=labels, dataset="knn", vline=True)

                os.makedirs("outputs", exist_ok=True)
                np.savez("outputs/knn_adj_{}_{}_{:.0f}.npz".format(dataset, seed, m2/m), data=knn_adj.data, row=knn_adj.row, col=knn_adj.col)

