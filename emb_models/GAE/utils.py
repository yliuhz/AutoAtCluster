
import scipy.sparse as sp
import torch
import numpy as np
import dgl

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

# return the (x,y)-coord of the idx-th element in a 2-dim matrix
def idx_to_coord(n, idx):
    x = idx / n
    y = idx % n
    return x, y

def convert_to_syn_adj(adj):
    adj = adj.tocoo()
    n = adj.shape[0]
    adj_t = adj.transpose()

    adj = (adj + adj_t).tocoo()
    row, col, data = adj.row, adj.col, adj.data

    data[data > 1] = 1
    adj = sp.coo_matrix((data, (row, col)), shape=(n,n))
    
    return adj


from scipy.linalg import fractional_matrix_power, inv
import scipy.sparse as sp


def compute_ppr(adj, alpha=0.2, self_loop=True):
    a = adj.toarray()
    if self_loop:
        a = a + np.eye(a.shape[0])                                # A^ = A + I_n
    d = np.diag(np.sum(a, 1))                                     # D^ = Sigma A^_ii
    dinv = fractional_matrix_power(d, -0.5)                       # D^(-1/2)
    at = np.matmul(np.matmul(dinv, a), dinv)                      # A~ = D^(-1/2) x A^ x D^(-1/2)
    return alpha * inv((np.eye(a.shape[0]) - (1 - alpha) * at))   # a(I_n-(1-a)A~)^-1

import argparse
def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cora', help='type of dataset.')
    parser.add_argument('--nexp', type=int, default=10, help="Number of repeated experiments")
    return parser


def louvain_cluster(adj, labels, random_state=None):
    from community import community_louvain
    import networkx as nx
    from sklearn.metrics import normalized_mutual_info_score as NMI, adjusted_mutual_info_score as AMI

    graph = nx.from_scipy_sparse_matrix(adj)
    partition = community_louvain.best_partition(graph, random_state=random_state)
    preds = list(partition.values())

    return preds

    ami = AMI(labels, preds)
    nmi = NMI(labels, preds)

    print("ami={:.3f}, nmi={:.3f}".format(ami, nmi))

