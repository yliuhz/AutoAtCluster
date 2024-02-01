
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
# from scipy.sparse.linalg.eigen.arpack import eigsh
from scipy.sparse.linalg import eigsh
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import networkit as nk

import sklearn.preprocessing as preprocess
from ogb.nodeproppred import DglNodePropPredDataset
import os

import time
from datetime import datetime

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')  # - %(name)s
logger = logging.getLogger(__name__)

def louvain_cluster(adj, labels, random_state=None):
    from community import community_louvain
    import networkx as nx
    from sklearn.metrics import normalized_mutual_info_score as NMI, adjusted_mutual_info_score as AMI

    graph = nx.from_scipy_sparse_array(adj)
    graph_nk = nk.nxadapter.nx2nk(graph, weightAttr="weight")

    alg_st = time.time()
    algo = nk.community.PLM(graph_nk, True)
    plmCommunities = nk.community.detectCommunities(graph_nk, algo=algo)
    alg_ed = time.time()

    preds = plmCommunities.getVector()

    return preds, alg_ed-alg_st

"""
Cluster unattributed graphs
    datapath: str
    graphFormat: choose from 
        networkit.graphio.Format.DOT
        networkit.graphio.Format.EdgeList
        networkit.graphio.Format.EdgeListCommaOne
        networkit.graphio.Format.EdgeListSpaceZero
        networkit.graphio.Format.EdgeListSpaceOne
        networkit.graphio.Format.EdgeListTabZero
        networkit.graphio.Format.EdgeListTabOne
        networkit.graphio.Format.GraphML
        networkit.graphio.Format.GraphToolBinary
        networkit.graphio.Format.GraphViz
        networkit.graphio.Format.GEXF
        networkit.graphio.Format.GML
        networkit.graphio.Format.KONECT
        networkit.graphio.Format.LFR
        networkit.graphio.Format.METIS
        networkit.graphio.Format.NetworkitBinary
        networkit.graphio.Format.SNAP
        networkit.graphio.Format.MatrixMarket
"""
def louvain_cluster_unattributed(datapath, graphFormat):
    graph_nk = nk.readGraph(datapath, graphFormat)

    alg_st = time.time()
    algo = nk.community.PLM(graph_nk, True)
    plmCommunities = nk.community.detectCommunities(graph_nk, algo=algo)
    alg_ed = time.time()

    preds = plmCommunities.getVector()

    return preds, alg_ed-alg_st

def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cora', help='type of dataset.')
    parser.add_argument('--nexp', type=int, default=10, help="Number of repeated experiments")
    return parser

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

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, required=True, choices=["all", "unweighted", "weighted", "queue"])
    parser.add_argument("--unattr", action="store_true")
    parser.add_argument("--datapath", type=str, default="")
    args = parser.parse_args()

    datasets = [
        # "cora",
        # "citeseer",
        # "wiki",
        "pubmed",
        # "amazon-photo",
        # "amazon-computers",
        # "cora-full",
        # "ogbn-arxiv"
    ]

    emb_paths = {
        "cora": "/home/yliumh/github/AutoAtCluster/emb_models/GGD/manual_version/outputs/GGD_{}_emb_{}.npz",
        "citeseer": "/home/yliumh/github/AutoAtCluster/emb_models/GGD/manual_version/outputs/GGD_{}_emb_{}.npz",
        "wiki": "/home/yliumh/github/AutoAtCluster/emb_models/AGE/outputs/AGE_{}_emb_{}.npz",
        "pubmed": "/home/yliumh/github/AutoAtCluster/emb_models/AGE/outputs/AGE_{}_emb_{}.npz",
        "amazon-photo": "/home/yliumh/github/AutoAtCluster/emb_models/GGD/manual_version/outputs/GGD_{}_emb_{}.npz",
        "amazon-computers": "/home/yliumh/github/AutoAtCluster/emb_models/GGD/manual_version/outputs/GGD_{}_emb_{}.npz",
        "cora-full": "/home/yliumh/github/graph2gauss/outputs/G2G_{}_emb_{}.npz",
        "ogbn-arxiv": "/home/yliumh/github/AutoAtCluster/emb_models/GGD/manual_version/outputs/GGD_{}_emb1_{}.npz"
    }

    # neighbor_modes = [
    #     "unweighted",
    #     "queue",
    #     "all",
    #     "weighted"
    # ]
    neighbor_modes = [
        f"{args.mode}"
    ]
    
    seeds = np.arange(3, dtype=int)

    os.makedirs("outputs", exist_ok=True)
    expdir = f"LouvainNKFull"
    if args.unattr:
        expdir = expdir + "-unattr"
    logger.info(f"Results stored in ./outputs/{expdir}")

    for neighbor_mode in neighbor_modes:

        if args.unattr:
            for seed in seeds:
                
                dataset = args.datapath.split("/")[-1].split(".")[0]
                logger.info(f"{neighbor_mode}, {dataset}, {seed}")
                setproctitle.setproctitle("LouvainNK-{}-{}-{}".format(neighbor_mode, dataset, seed))

                if os.path.exists("outputs/{}/Louvain_{}_{}_{}.npz".format(expdir, dataset, seed, neighbor_mode)):
                    logger.info(f"Skip: unattr_adj_{dataset}_{seed}_{neighbor_mode}.npz")
                    continue

                try:
                    func_st = time.time()
                    preds, alg_time = louvain_cluster_unattributed(args.datapath, nk.Format.EdgeListTabZero)
                    func_end = time.time()

                    time_now = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")

                    os.makedirs(f"outputs/{expdir}", exist_ok=True)
                    with open(f"outputs/{expdir}/time.txt", "a+") as f:
                        f.write(f"{dataset}\t{seed}\t{neighbor_mode}\t{alg_time}\t{func_end-func_st}\t{time_now}\n")
                    np.savez("outputs/{}/Louvain_{}_{}_{}.npz".format(expdir, dataset, seed, neighbor_mode), preds=preds)
                except Exception as e:
                    logger.error(f"Error: unattr_adj_{dataset}_{seed}_{neighbor_mode}.npz {e}")
                else:
                    logger.info(f"Success: unattr_adj_{dataset}_{seed}_{neighbor_mode}.npz")

        else:
            for dataset in datasets:
                
                adj, features, labels = load_data(dataset)

                n = adj.shape[0]
                m = adj.sum()

                for seed in seeds:
                    logger.info(f"{neighbor_mode}, {dataset}, {seed}")

                    np.random.seed(seed)
                    random.seed(seed)

                    setproctitle.setproctitle("LouvainNKFull-{}-{}-{}".format(neighbor_mode, dataset, seed))

                    if os.path.exists("outputs/{}/Louvain_{}_{}_{}.npz".format(expdir, dataset, seed, neighbor_mode)):
                        logger.info(f"Skip: full_adj_{dataset}_{seed}_{neighbor_mode}.npz")
                        continue

                    try:
                        emb_path = emb_paths[dataset].format(dataset, seed)
                        logger.info(f"Embedding path= {emb_path}")
                        data = np.load(emb_path)
                        emb = data["emb"]
                        emb = torch.FloatTensor(emb)
                        emb = F.normalize(emb)
                        sim = torch.mm(emb, emb.t())
                        sim = F.relu(sim).numpy()

                        sim_row = np.repeat(np.arange(n), n) #000111222
                        sim_col = np.tile(np.arange(n), n) #012012012
                        sim_data = sim.flatten()

                        full_adj = sp.coo_matrix((sim_data, (sim_row, sim_col)), shape=(n,n))


                        # knn_adj = knn_graph(emb, k, non_linear, i=6)
                        func_st = time.time()
                        preds, alg_time = louvain_cluster(full_adj, labels, random_state=seed)
                        func_end = time.time()

                        time_now = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")

                        os.makedirs(f"outputs/{expdir}", exist_ok=True)
                        with open(f"outputs/{expdir}/time.txt", "a+") as f:
                            f.write(f"{dataset}\t{seed}\t{neighbor_mode}\t{alg_time}\t{func_end-func_st}\t{time_now}\n")
                        np.savez("outputs/{}/Louvain_{}_{}_{}.npz".format(expdir, dataset, seed, neighbor_mode), preds=preds, labels=labels)
                    except Exception as e:
                        logger.error(f"Error: full_adj_{dataset}_{seed}_{neighbor_mode}.npz {e}")
                    else:
                        logger.info(f"Success: full_adj_{dataset}_{seed}_{neighbor_mode}.npz")

                