import argparse
import os.path as osp
import random
from time import perf_counter as t
import yaml
from yaml import SafeLoader

import torch
import torch_geometric.transforms as T
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.datasets import Planetoid, CitationFull
from torch_geometric.utils import dropout_adj
from torch_geometric.nn import GCNConv
import numpy as np
from torch_geometric.data import Data

import scipy.sparse as sp
import sklearn.preprocessing as preprocess

from model import Encoder, Model, drop_feature
from eval import label_classification

import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh
import sys
import torch
import torch.nn as nn

import sklearn.preprocessing as preprocess
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score as NMI

import setproctitle


def train(model: Model, x, edge_index):
    model.train()
    optimizer.zero_grad()
    edge_index_1 = dropout_adj(edge_index, p=drop_edge_rate_1)[0]
    edge_index_2 = dropout_adj(edge_index, p=drop_edge_rate_2)[0]
    x_1 = drop_feature(x, drop_feature_rate_1)
    x_2 = drop_feature(x, drop_feature_rate_2)
    z1 = model(x_1, edge_index_1)
    z2 = model(x_2, edge_index_2)

    loss = model.loss(z1, z2, batch_size=0)
    loss.backward()
    optimizer.step()

    return loss.item()


def test(model: Model, x, edge_index, y, final=False):
    model.eval()
    z = model(x, edge_index)

    label_classification(z, y, ratio=0.1)

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
            with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
                if sys.version_info > (3, 0):
                    objects.append(pkl.load(f, encoding='latin1'))
                else:
                    objects.append(pkl.load(f))

        x, y, tx, ty, allx, ally, graph = tuple(objects)
        test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
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

        features = features.toarray()


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
    f = open('data/graph.txt', 'r')
    adj, xind, yind = [], [], []
    for line in f.readlines():
        line = line.split()

        xind.append(int(line[0]))
        yind.append(int(line[1]))
        adj.append([int(line[0]), int(line[1])])
    f.close()
    ##print(len(adj))

    f = open('data/group.txt', 'r')
    label = []
    for line in f.readlines():
        line = line.split()
        label.append(int(line[1]))
    f.close()

    f = open('data/tfidf.txt', 'r')
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

    # features = sp.csr_matrix(features)


    # label = torch.FloatTensor(label)
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


class Dataset(object):
    def __init__(self, data):
        self.dataset = [data]
        self.num_features = data.x.shape[-1]
    def __getitem__(self, item):
        return self.dataset[item]

def convert2Dataset(adj, features, labels):
    features = torch.FloatTensor(features)
    labels = torch.LongTensor(labels)

    edge_index = torch.LongTensor(np.array(adj.nonzero()))
    data = Data(x=features, edge_index=edge_index, y=labels)
    return Dataset(data)


def eval(model, x, edge_index, nclass=8):
    with torch.no_grad():
        model.eval()
        z = model(x, edge_index).cpu().numpy()

        cluster = KMeans(n_clusters=nclass)
        preds = cluster.fit_predict(z)
        nmi = NMI(data.y.cpu().numpy(), preds)

    return nmi

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='DBLP')
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument("--nexp", default=10, type=int)
    args = parser.parse_args()

    assert args.gpu_id in range(0, 8)
    torch.cuda.set_device(args.gpu_id)

    config = yaml.load(open(args.config), Loader=SafeLoader)[args.dataset]

    # torch.manual_seed(config['seed'])
    # random.seed(12345)

    learning_rate = config['learning_rate']
    num_hidden = config['num_hidden']
    num_proj_hidden = config['num_proj_hidden']
    activation = ({'relu': F.relu, 'prelu': nn.PReLU()})[config['activation']]
    base_model = ({'GCNConv': GCNConv})[config['base_model']]
    num_layers = config['num_layers']

    drop_edge_rate_1 = config['drop_edge_rate_1']
    drop_edge_rate_2 = config['drop_edge_rate_2']
    drop_feature_rate_1 = config['drop_feature_rate_1']
    drop_feature_rate_2 = config['drop_feature_rate_2']
    tau = config['tau']
    num_epochs = config['num_epochs']
    weight_decay = config['weight_decay']

    def get_dataset(path, name):
        assert name in ['Cora', 'CiteSeer', 'PubMed', 'DBLP', "Wiki", "amazon-photo", "amazon-computers", "cora-full"]
        name = 'dblp' if name == 'DBLP' else name

        # if name == "Wiki":
        #     return load_wiki()

        if name in ['Cora', 'CiteSeer', 'PubMed', "Wiki", "amazon-photo", "amazon-computers"]:
            name = name.casefold()
            adj, features, labels, _, _, _ = load_data(name)
            return convert2Dataset(adj, features, labels)



        return (CitationFull if name == 'dblp' else Planetoid)(
            path,
            name,
            transform = T.NormalizeFeatures())

    # path = osp.join(osp.expanduser('~'), 'datasets', args.dataset)
    path = osp.join(".", "data")
    dataset = get_dataset(path, args.dataset)

    seeds = np.arange(0, args.nexp, dtype=int)

    for seed in seeds:
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        random.seed(seed)

        setproctitle.setproctitle("GRACE-{}-{}".format(args.dataset[:2], seed))

        data = dataset[0]

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        data = data.to(device)

        encoder = Encoder(dataset.num_features, num_hidden, activation,
                        base_model=base_model, k=num_layers).to(device)
        model = Model(encoder, num_hidden, num_proj_hidden, tau).to(device)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        start = t()
        prev = start
        best_nmi = 0.0
        for epoch in range(1, num_epochs + 1):
            loss = train(model, data.x, data.edge_index)
            
            # if epoch % 100 == 1:
            #     nmi = eval(model, data.x, data.edge_index, nclass=10)
            #     best_nmi = max(nmi, best_nmi)

            now = t()
            print(f'(T) | Epoch={epoch:03d}, loss={loss:.4f}, '
                f'this epoch {now - prev:.4f}, total {now - start:.4f}')
                # f'NMI {nmi:.4f}')
            prev = now

        print("=== Final ===")
        # test(model, data.x, data.edge_index, data.y, final=True)

        model.eval()
        with torch.no_grad():
            z = model(data.x, data.edge_index).cpu().numpy()

            import os
            os.makedirs("outputs", exist_ok=True)
            np.savez("outputs/GRACE_{}_emb_{}.npz".format(args.dataset, seed), emb=z)

            # from sklearn.cluster import KMeans
            # from sklearn.metrics import normalized_mutual_info_score as NMI
            # cluster = KMeans(n_clusters=8)
            # preds = cluster.fit_predict(z)
            # nmi = NMI(data.y.cpu().numpy(), preds)

            # print("NMI({})={:.3f}".format(args.dataset, nmi))
            # print("Best NMI({})={:.3f}".format(args.dataset, best_nmi))
