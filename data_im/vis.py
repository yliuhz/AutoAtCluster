from locale import normalize
import os
import time
import argparse
from turtle import pos
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
from dgl import DGLGraph
import dgl.function as fn

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)

# torch.cuda.set_device(0)

import dgl
from tqdm import tqdm
from mpl_toolkits.axes_grid1 import make_axes_locatable

import itertools
import scipy.sparse as sp

# csr_matrix: 行内indices动, data不动; 行间indices动, data动
# sparse_matrix: n * n
def reorder_sparse(sparse_matrix, new_idx):
    indices, indptr, data = sparse_matrix.indices, sparse_matrix.indptr, sparse_matrix.data
    n = sparse_matrix.shape[0]
    new_idx = np.array(new_idx)
    new_idx_s = np.argsort(new_idx)
    indices_l, data_l = [], []
    for i in range(n):
        indices_l.append(indices[indptr[i]:indptr[i+1]].tolist())
        data_l.append(data[indptr[i]:indptr[i+1]].tolist())
    # 先行内动
    for i in range(n):
        temp = [new_idx_s[j] for j in indices_l[i]]
        indices_l[i] = temp
    
    # 再行间动
    indices_l = [indices_l[i] for i in new_idx]
    data_l = [data_l[i] for i in new_idx]
    new_indptr = [0]
    for i in range(n):
        new_indptr.append(new_indptr[i]+len(indices_l[i]))
    new_indices = list(itertools.chain(*indices_l))
    new_data = list(itertools.chain(*data_l))

    ret = sp.csr_matrix((new_data, new_indices, new_indptr), shape=(n,n)).sorted_indices()
    return ret

def reorder_sparse_coo(sparse_matrix, new_idx):
    sparse_matrix = sparse_matrix.tocoo()
    row, col, data = sparse_matrix.row, sparse_matrix.col, sparse_matrix.data
    n = sparse_matrix.shape[0]
    new_idx = np.array(new_idx)
    new_idx_s = np.argsort(new_idx)

    row, col = new_idx_s[row], new_idx_s[col]

    return sp.coo_matrix((data, (row, col)), shape=(n,n))



def superPixels(adj, K=100, sparse=False):
    n = adj.shape[0]
    unit = n // K
    superadj = np.zeros((K, K))
    if not sparse:
        print("N sparse")
        for i in range(K):
            for j in range(K):
                temp = adj[(i*unit):min((i+1)*unit, n), (j*unit):min((j+1)*unit, n)]
                superadj[i, j] = temp.mean()
    else:
        adj = adj.tocsr()
        indices, indptr, data = adj.indices, adj.indptr, adj.data
        for i in tqdm(range(K), total=K):
            for j in range(K):
                row_min = i*unit
                row_max = min((i+1)*unit, n)
                col_min = j*unit
                col_max = min((j+1)*unit, n)
                
                temp = 0.

                for row in range(row_min, row_max):
                    indices_ = indices[indptr[row]:indptr[row+1]]
                    if len(indices_) > 0 and indices_[-1] >= col_min and indices_[0] < col_max:
                        st = np.searchsorted(indices_, col_min)
                        ed = np.searchsorted(indices_, col_max)
                        # st += indptr[row]
                        # ed += indptr[row]
                        # temp += data[st:ed].sum()
                        temp += ed - st
                        # for k in range(indptr[row], indptr[row+1]):
                        #     if indices[k] >= col_min and indices[k] < col_max:
                        #         temp += data[k]
                        #         cnt += 1

                temp /= (row_max - row_min) * (col_max - col_min)

                superadj[i, j] = temp
    return superadj

def plot_superadj(adj, K=100, sparse=True, labels=None, dataset="", vline=False):
    if labels is not None:
        idx = np.argsort(labels, 0)
        if sparse:
            # adj = reorder_sparse(adj, idx)
            adj = reorder_sparse_coo(adj, idx)
        else:
            adj = adj[idx, :][:, idx]
    n = adj.shape[0]

    adj = superPixels(adj, K, sparse=sparse)
    plt.figure()
    ax = plt.gca()
    plt.imshow(adj, cmap="coolwarm")
    plt.colorbar()
    if labels is not None and vline:
        labels = np.array(labels, dtype=int)
        labels = np.sort(labels)
        minl, maxl = labels[0], labels[-1]
        xs = []
        for l in range(minl+1, maxl+1):
            idx = np.searchsorted(labels, l)
            xs.append(idx)
        unit = n // K
        xs = [x / unit for x in xs]
        for x in xs:
            plt.axvline(x=x, color="red")
    # plt.clim(0.0, 0.04)
    os.makedirs("pics", exist_ok=True)
    plt.savefig("pics/adj_{}.png".format(dataset))
