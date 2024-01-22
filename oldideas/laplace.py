
import numpy as np
from sklearn import metrics
import scipy.sparse as sp
from sklearn.cluster import SpectralClustering
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import os

def laplace(adj, features, layer=4, renorm=True):
    adj = sp.coo_matrix(adj)
    ident = sp.eye(adj.shape[0])
    sm_fea_s = sp.csr_matrix(features).toarray()

    if renorm:
        adj_ = adj + ident
    else:
        adj_ = adj
    
    rowsum = np.array(adj_.sum(1))

    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    laplacian = ident - adj_normalized

    # layer = 4
    maxdb = 0.0
    a = ident - laplacian

    for i in tqdm(range(layer)):
        sm_fea_s = a.dot(sm_fea_s)

    return sm_fea_s


from dcsbm import DCSBM
from heuristic import Heuristic
from laplace import laplace
from vis import plot_superadj
from model import GAE, DGI, MVGRL

import numpy as np
from sklearn.cluster import SpectralClustering, KMeans, MiniBatchKMeans
from community import community_louvain
from sklearn.metrics import normalized_mutual_info_score as NMI
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp

from matplotlib import pyplot as plt
import os
from tqdm import tqdm

from load import load_assortative, load_syn_h2gcn, preprocess_graph, load_disassortative, preprocess_features
from utils import sparse_mx_to_torch_sparse_tensor, compute_ppr

import dgl
import dgl.nn as dnn

if __name__ == "__main__":
    K = 2
    n = 2000
    seed=20
    p = 1.0 # wlm / wkk
    w0 = n # wkk
    D = 512
    hidden_dim = 128 # 32
    emb_dim = 32 # 16
    gnn_layers = 2
    device = "cuda:0"
    epochs = 200
    lr=0.001
    wd = 5e-4
    batch_size = 1
    b_xent = nn.BCEWithLogitsLoss()
    sample_size = 2000

    np.random.seed(seed)
    torch.manual_seed(seed)

    c = np.ones(K, dtype=int) * (n // K)

    nmis = []

    ps = np.arange(0, 1.1, 0.1)
    ps = [0.0]
    for p in ps:
        # print("p={}".format(p))

        # w = np.eye(K, dtype=int) * n
        # w = (1 - np.eye(K, dtype=int)) * (n // 4) # multi-partite
        # # w = [[0, 2 * n],[2 * n, n * 5]] # core-periphery
        # # w = np.eye(K, dtype=int) * w0 * p + (1 - np.eye(K, dtype=int)) * w0 * 12 * (1-p) # mixing

        # theta = np.random.rand(n)
        # c_sum = np.cumsum(c)
        # for i in range(K):
        #     if i == 0:
        #         low = 0
        #     else:
        #         low = c_sum[i-1]
        #     theta_ = theta[low:c_sum[i]]

        #     theta_ = theta_ / theta_.sum()
        #     theta[low:c_sum[i]] = theta_
        
        # lamb = 0.8
        
        # dataset = "syn"
        # dataset = "cora"
        dataset = "fb100-penn94"
        # dataset = "genius"
        # dataset = "citeseer"
        K = 2
        print("dataset={} {}".format(dataset, K))
        nmis_r = []
        for r in range(1, 2):
            print('Generating graphs ...')
            # dataset = "cora"
            # dcsbm = DCSBM()
            # adj, features, labels = dcsbm.generate(c, w, theta, lamb, random_state=seed, save=False)
            # adj, features, labels = load_assortative(dataset=dataset)
            # adj, features, labels = load_syn_h2gcn(dataset=dataset, h=p, r=r)
            adj, features, labels = load_disassortative(dataset=dataset)
            # features = preprocess_features(features)
            adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
            adj.eliminate_zeros()
            n, m, d = adj.shape[0], adj.sum(), features.shape[1]
            print("[nodes,edges,features]=[{}, {}, {}]".format(n, m, d))
            adj_norm = preprocess_graph(adj, renorm=True)
            adj_1st = adj + sp.eye(n)
            # adj_1st = (adj + sp.eye(n)).toarray()
            # adj_1st = sparse_mx_to_torch_sparse_tensor(adj + sp.eye(n))
            # plot_superadj(adj, K=min(adj.shape[0], 100), dataset=dataset, sparse=True, labels=labels, vline=True)



            # g = dgl.from_scipy(adj)
            # g = dgl.to_simple(g)
            # g = dgl.remove_self_loop(g)
            # g = dgl.to_bidirected(g)
            # g = dgl.add_self_loop(g)

            # ids = np.arange(0, n)
            # zero_node_idx = ids[(labels == 0)]
            # one_node_idx = ids[(labels == 1)]
            # print(zero_node_idx.shape, one_node_idx.shape)
            # print(zero_node_idx[:5], one_node_idx[:5])
            # z01 = adj.getrow(zero_node_idx[0]).toarray()
            # z02 = adj.getrow(zero_node_idx[1]).toarray()
            # z1 = adj.getrow(one_node_idx[0]).toarray()

            # def cosine_s(z1, z2):
            #     z1 = normalize(z1)
            #     z2 = normalize(z2)
            #     s = np.matmul(z1, np.transpose(z2))
            #     return s[0,0]

            # def product(z1, z2):
            #     s = np.matmul(z1, np.transpose(z2))
            #     return s[0,0]

            # adj = adj.toarray()[zero_node_idx, :][:, zero_node_idx]
            # print(adj.shape)
            # adj = normalize(adj)
            # s = np.matmul(adj, np.transpose(adj))
            # s = 



            # exit(0)



            print("Heuristic ...")
            heu = Heuristic()
            # adj_ = heu.detect3(adj_norm)
            saved_new_adj = True
            if not saved_new_adj:
                saved_sim = True
                if not saved_sim:
                    adj_ = heu.detect3(adj_1st)
                    np.savez("outputs/sim_{}.npz".format(dataset), row=adj_.row, col=adj_.col, data=adj_.data)
                else:
                    data_ = np.load("outputs/sim_{}.npz".format(dataset))
                    row, col, data = data_["row"], data_["col"], data_["data"]
                    adj_ = sp.coo_matrix((data, (row, col)), shape=(n,n))
                    adj_.eliminate_zeros()
                # # NOTE: 为可视化重排节点
                # idx = np.argsort(labels, 0)
                # adj_ = adj_[idx, :][:, idx]
                # labels = labels[idx]
                # features = sp.csr_matrix(features.toarray()[idx, :])
                # plot_superadj(adj_.tocsr(), K=min(n, 100), dataset="heuristic_1", sparse=True, labels=labels, vline=True)
                thres = heu.set_threshold_nano_sparse(adj_, K=K)
                adj_ = heu.normalize_adj(adj_, thres)
                print('thres={}'.format(thres))
                plot_superadj(adj_, K=min(n, 100), dataset="heuristic", sparse=True, labels=labels, vline=True)
                np.savez("outputs/newadj_{}.npz".format(dataset), row=adj_.row, col=adj_.col, data=adj_.data)
            else:
                data_ = np.load("outputs/newadj_{}.npz".format(dataset))
                row, col, data = data_["row"], data_["col"], data_["data"]
                adj_ = sp.coo_matrix((data, (row, col)), shape=(n,n))
                adj_.eliminate_zeros()
                # plot_superadj(adj_, K=min(n, 100), dataset="heuristic", sparse=True, labels=labels, vline=True)
            adj = sp.coo_matrix(adj_)
            adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
            adj.eliminate_zeros()
            n, m, d = adj.shape[0], adj.sum(), features.shape[1]
            print("H-[nodes,edges,features]=[{}, {}, {}]".format(n, m, d))
            adj_norm = preprocess_graph(adj, renorm=True)
            adj_1st = (adj + sp.eye(n)).toarray()


            print('Preparing the model ...')
            
            emb = laplace(adj, features, layer=20, renorm=True)
            # emb = features.toarray()

            print("End of training.")
            print('Clustering ...')
            os.makedirs("outputs", exist_ok=True)
            np.savez("outputs/{}_emb.npz".format(dataset), emb=emb, labels=labels)
            # emb = normalize(emb)
            # f_adj = np.matmul(emb, np.transpose(emb))
            # print("mean={}, max={}, min={}".format(f_adj.mean(), f_adj.max(), f_adj.min()))
            # cluster = KMeans(n_clusters=K, random_state=10)
            cluster = MiniBatchKMeans(n_clusters=K, random_state=0, batch_size=1024, verbose=0)
            # cluster = SpectralClustering(n_clusters=K, random_state=0)
            preds = cluster.fit_predict(emb)
            nmi = NMI(labels, preds)
            nmis_r.append(nmi)

            # print('Plotting ...')
            # tsne_z = TSNE(n_components=2, init="random").fit_transform(emb)
            # plt.figure(figsize=(10,10))
            # plt.scatter(tsne_z[:,0], tsne_z[:,1], c=labels)
            # plt.savefig("pics/cluster.png")

        nmis.append(np.mean(nmis_r))
        print("nmi={:.3f}".format(np.mean(nmis_r)))


    os.makedirs("results", exist_ok=True)
    with open("results/results.txt", 'a+') as f:
        for nmi in nmis:
            f.write('{} '.format(nmi))
        f.write("\n\n")







# if __name__ == "__main__":
#     seed=20

#     filename = "data/dcsbm_a.npz"
#     data = np.load(filename)
#     adj, labels = data["adj"], data["labels"]
#     K = np.unique(labels).shape[0]
#     n = adj.shape[0]
#     print('n={}, K={}'.format(n, K))

#     # adj = 1. - adj
#     # peru = np.random.RandomState(seed=seed).permutation(n)
#     # adj = adj[peru,:][:,peru]
#     features = np.eye(n)
#     # features = features[peru]
#     features = laplace(adj, features)

#     # f_adj = np.matmul(features, np.transpose(features))
#     # cluster = SpectralClustering(n_clusters=K, affinity="precomputed", random_state=0)
#     # preds = cluster.fit_predict(f_adj)


#     tsne_z = TSNE(n_components=2, init="random").fit_transform(features)
#     plt.figure()
#     plt.scatter(tsne_z[:,0], tsne_z[:,1], c=labels)
#     os.makedirs("pics", exist_ok=True)
#     plt.savefig("pics/tsne_laplace.png")



