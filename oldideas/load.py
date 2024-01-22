import numpy as np
import os
import os.path as osp
from deeprobust.graph.data import Dataset
import scipy.sparse as sp
import scipy
import torch
from sklearn.preprocessing import label_binarize
from sklearn import preprocessing

from utils import convert_to_syn_adj
from tqdm import tqdm

from ogb.nodeproppred import DglNodePropPredDataset

# 统一返回: adj: csr_matrix, features: csr_matrix, labels: numpy.darray


# real dataset: cora, citeseer, pubmed
def load_assortative(dataset="cora"):
    import pickle as pkl
    import networkx as nx
    import scipy.sparse as sp
    import torch

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
        ##print(len(adj))

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

        return adj, features, labels
    else:
        raise NotImplementedError()

def load_disassortative(dataset="fb100-penn94"):
    assert dataset in ["fb100-penn94", "genius"], "Unknown dataset"

    if dataset == "fb100-penn94":
        mat = scipy.io.loadmat('data/fb100-penn94/Penn94.mat')
        A = mat['A']
        metadata = mat['local_info']

        edge_index = torch.tensor(A.nonzero(), dtype=torch.long)
        metadata = metadata.astype(np.int)
        label = metadata[:, 1] - 1  # gender label, -1 means unlabeled
        feature_vals = np.hstack(
            (np.expand_dims(metadata[:, 0], 1), metadata[:, 2:]))
        features = np.empty((A.shape[0], 0))
        for col in range(feature_vals.shape[1]):
            feat_col = feature_vals[:, col]
            feat_onehot = label_binarize(feat_col, classes=np.unique(feat_col))
            features = np.hstack((features, feat_onehot))

        node_feat = torch.tensor(features, dtype=torch.float)
        num_nodes = metadata.shape[0]

        n, m = num_nodes, edge_index.shape[1]
        edge_index = edge_index.numpy()

        labels = torch.tensor(label)
        if labels.ndim > 1:
            if labels.shape[1] == 1:
                labels = labels.view(-1)
            else:
                labels = labels.argmax(1)
        labels = labels.numpy()

        CUT = True
        if CUT:
            print('Cuting ...')
            nn = np.arange(n, dtype=int)
            u_idx = nn[labels == -1]
            u_idx = set(u_idx.tolist())
            k_idx = nn[labels >= 0]
            k_idx_s = {v:k for k, v in enumerate(k_idx)}

            data_ = np.ones(m)

            for i in range(m):
                u, v = edge_index[0,i], edge_index[1,i]
                if u in u_idx or v in u_idx:
                    data_[i] = 0
                else:
                    edge_index[0,i] = k_idx_s[u]
                    edge_index[1,i] = k_idx_s[v]

            # print(n, edge_index[0].max(), edge_index[1].max())
            adj = sp.coo_matrix((data_, (edge_index[0], edge_index[1])), shape=(n,n))
            adj.eliminate_zeros()
            n = k_idx.shape[0]
            data_, row, col = adj.data, adj.row, adj.col
            adj = sp.coo_matrix((data_, (row, col)), shape=(n,n))
            features = sp.coo_matrix(node_feat.numpy()[k_idx])
            labels = labels[k_idx]
        else:
            adj = sp.coo_matrix((np.ones(m), (edge_index[0], edge_index[1])), shape=(n,n))
            features = sp.coo_matrix(node_feat.numpy())

        # adj = adj.toarray()
        # n = adj.shape[0]
        # nn = np.arange(0, n, dtype=int)
        # zero_idx = nn[(labels == 0)]
        # one_idx = nn[(labels == 1)][:zero_idx.shape[0]]
        # idx = np.transpose([np.tile(one_idx, len(one_idx)), np.repeat(one_idx, len(one_idx))])
        # idx_ = idx[:, 0] * n + idx[:, 1]
        # np.put(adj, idx_, adj[zero_idx, :][:, zero_idx])
        # adj = sp.coo_matrix(adj)

        return adj, features, labels
    
    elif dataset == "genius":
        fulldata = scipy.io.loadmat(f'data/genius/genius.mat')

        edge_index = torch.tensor(fulldata['edge_index'], dtype=torch.long)
        node_feat = torch.tensor(fulldata['node_feat'], dtype=torch.float)
        label = torch.tensor(fulldata['label'], dtype=torch.long).squeeze()
        num_nodes = label.shape[0]

        n, m = num_nodes, edge_index.shape[1]

        adj = sp.coo_matrix((np.ones(m), (edge_index[0].numpy(), edge_index[1].numpy())), shape=(n,n))
        adj = convert_to_syn_adj(adj)
        features = sp.coo_matrix(node_feat.numpy())
        
        labels = torch.tensor(label)
        if labels.ndim > 1:
            if labels.shape[1] == 1:
                labels = labels.view(-1)
            else:
                labels = labels.argmax(1)
        labels = labels.numpy()

        return adj, features, labels

def load_cora_full_im(rate=0.1, seed=None):
    filename = "/data/liuyue/New/SBM/mySBM/data_im/cora-full_{:.1f}_{:d}_5.npz".format(rate, seed)
    data = np.load(filename)

    adj_raw, features_raw, labels_raw = load_assortative("cora-full")

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

    adj_raw, features_raw, labels_raw = load_assortative("cora-full")

    adj_data, adj_row, adj_col, features_load, labels_load, mask = data["data"], data["row"], data["col"], data["features"], data["labels"], data["mask"]
    adj_load = sp.coo_matrix((adj_data, (adj_row, adj_col)), shape=(labels_load.shape[0], labels_load.shape[0]))

    adj_mask = adj_raw.toarray()[mask,:][:,mask]
    assert (adj_mask - adj_load).sum() < 1e-7
    features_mask = features_raw.toarray()[mask]
    assert (features_mask - features_load).sum() < 1e-7

    return adj_load, features_load, labels_load, mask


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

    return adj_load, features_load, labels_load, mask

def load_obgn_arxiv_kd(rate=0.1, seed=0):
    filename = "/home/yliumh/github/AutoAtCluster/data_im/dataset/ogbn-arxiv_{:.1f}_{:d}_kd.npz".format(rate, seed)
    data = np.load(filename)

    # adj_raw, features_raw, labels_raw = load_assortative("cora-full")

    adj_data, adj_row, adj_col, features_load, labels_load, mask = data["data"], data["row"], data["col"], data["features"], data["labels"], data["mask"]
    adj_load = sp.coo_matrix((adj_data, (adj_row, adj_col)), shape=(labels_load.shape[0], labels_load.shape[0]))

    # adj_mask = adj_raw.toarray()[mask,:][:,mask]
    # assert (adj_mask - adj_load).sum() < 1e-7
    # features_mask = features_raw.toarray()[mask]
    # assert (features_mask - features_load).sum() < 1e-7

    return adj_load, features_load, labels_load


# load H2GCN synthetic dataset: syn-cora, syn-product
def load_syn_h2gcn(dataset="cora", h=0.00, r=1):
    datasetroot="/data/liuyue/New/H2GCN/npz-datasets/"
    datasetdir="syn-{}".format(dataset)
    datasetname="h{:.2f}-r{:d}".format(h, r)
    datasetsuffix=".npz"

    datasetpath = os.path.join(datasetroot, datasetdir, datasetname+datasetsuffix)

    assert os.path.exists(datasetpath), "{} not exist".format(datasetpath)

    dataset = CustomDataset(root=os.path.join(datasetroot, datasetdir), name=datasetname, setting="gcn", seed=15)
    adj = dataset.adj # Access adjacency matrix, csr_matrix
    features = dataset.features # Access node features, csr_matrix
    labels = dataset.labels # numpy.darray

    return adj, features, labels


def load_syn_subgraph(dataset="cora"):
    data = np.load("data/{}_subgraph/graph.npz".format(dataset))

    adj, features, labels = data["adj"], data["features"], data["labels"]

    adj = sp.coo_matrix(adj)
    features = sp.coo_matrix(features)

    return adj, features, labels


# return Laplacian matrix of adj
def preprocess_graph(adj, renorm=True):
    adj = sp.coo_matrix(adj)
    ident = sp.eye(adj.shape[0])
    if renorm:
        adj_ = adj + ident
    else:
        adj_ = adj
    
    rowsum = np.array(adj_.sum(1))
    
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    laplacian = ident - adj_normalized
        
    return adj_normalized

def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features


class CustomDataset(Dataset):
    def __init__(self, root, name, setting='gcn', seed=None, require_mask=False):
        '''
        Adopted from https://github.com/DSE-MSU/DeepRobust/blob/master/deeprobust/graph/data/dataset.py
        '''
        self.name = name.lower()
        self.setting = setting.lower()

        self.seed = seed
        self.url = None
        self.root = osp.expanduser(osp.normpath(root))
        self.data_folder = osp.join(root, self.name)
        self.data_filename = self.data_folder + '.npz'
        # Make sure dataset file exists
        assert osp.exists(self.data_filename), f"{self.data_filename} does not exist!" 
        self.require_mask = require_mask

        self.require_lcc = True if setting == 'nettack' else False
        self.adj, self.features, self.labels = self.load_data()
        self.idx_train, self.idx_val, self.idx_test = self.get_train_val_test()
        if self.require_mask:
            self.get_mask()
    
    def get_adj(self):
        adj, features, labels = self.load_npz(self.data_filename)
        adj = adj + adj.T
        adj = adj.tolil()
        adj[adj > 1] = 1

        if self.require_lcc:
            lcc = self.largest_connected_components(adj)

            # adj = adj[lcc][:, lcc]
            adj_row = adj[lcc]
            adj_csc = adj_row.tocsc()
            adj_col = adj_csc[:, lcc]
            adj = adj_col.tolil()

            features = features[lcc]
            labels = labels[lcc]
            assert adj.sum(0).A1.min() > 0, "Graph contains singleton nodes"

        # whether to set diag=0?
        adj.setdiag(0)
        adj = adj.astype("float32").tocsr()
        adj.eliminate_zeros()

        assert np.abs(adj - adj.T).sum() == 0, "Input graph is not symmetric"
        assert adj.max() == 1 and len(np.unique(adj[adj.nonzero()].A1)) == 1, "Graph must be unweighted"

        return adj, features, labels
    
    def get_train_val_test(self):
        if self.setting == "exist":
            with np.load(self.data_filename) as loader:
                idx_train = loader["idx_train"]
                idx_val = loader["idx_val"]
                idx_test = loader["idx_test"]
            return idx_train, idx_val, idx_test
        else:
            return super().get_train_val_test()

# Copied code from Non-homophily-large-scale
class NCDataset(object):
    def __init__(self, name, root=f''):
        """
        based off of ogb NodePropPredDataset
        https://github.com/snap-stanford/ogb/blob/master/ogb/nodeproppred/dataset.py
        Gives torch tensors instead of numpy arrays
            - name (str): name of the dataset
            - root (str): root directory to store the dataset folder
            - meta_dict: dictionary that stores all the meta-information about data. Default is None, 
                    but when something is passed, it uses its information. Useful for debugging for external contributers.
        
        Usage after construction: 
        
        split_idx = dataset.get_idx_split()
        train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
        graph, label = dataset[0]
        
        Where the graph is a dictionary of the following form: 
        dataset.graph = {'edge_index': edge_index,
                         'edge_feat': None,
                         'node_feat': node_feat,
                         'num_nodes': num_nodes}
        For additional documentation, see OGB Library-Agnostic Loader https://ogb.stanford.edu/docs/nodeprop/
        """

        self.name = name  # original name, e.g., ogbn-proteins
        self.graph = {}
        self.label = None

    # def get_idx_split(self, split_type='random', train_prop=.5, valid_prop=.25):
    #     """
    #     train_prop: The proportion of dataset for train split. Between 0 and 1.
    #     valid_prop: The proportion of dataset for validation split. Between 0 and 1.
    #     """

    #     if split_type == 'random':
    #         ignore_negative = False if self.name == 'ogbn-proteins' else True
    #         train_idx, valid_idx, test_idx = rand_train_test_idx(
    #             self.label, train_prop=train_prop, valid_prop=valid_prop, ignore_negative=ignore_negative)
    #         split_idx = {'train': train_idx,
    #                      'valid': valid_idx,
    #                      'test': test_idx}
    #     return split_idx

    def __getitem__(self, idx):
        assert idx == 0, 'This dataset has only one graph'
        return self.graph, self.label

    def __len__(self):
        return 1

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, len(self))


if __name__ == "__main__":
    for rate in np.arange(0.1, 1.0, 0.1):
        print(rate)
        adj, features, labels, mask = load_cora_full_im(rate=rate)
    