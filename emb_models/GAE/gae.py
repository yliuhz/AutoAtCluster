

from model import GAE, DGI, MVGRL, VGAEModel

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
from utils import sparse_mx_to_torch_sparse_tensor, compute_ppr, make_parser

import dgl
import dgl.nn as dnn

import setproctitle

if __name__ == "__main__":
    parser = make_parser()
    parser.add_argument("--model", type=str, default="GAE")
    args = parser.parse_args()

    K = 2
    n = 2000
    seed=20
    p = 1.0 # wlm / wkk
    w0 = n # wkk
    D = 512
    hidden_dim = 32 # 32
    emb_dim = 16 # 16
    gnn_layers = 2
    device = "cuda:0"
    epochs = 200
    lr=0.01
    wd = 0.0
    batch_size = 1
    b_xent = nn.BCEWithLogitsLoss()
    sample_size = 2000

    # np.random.seed(seed)
    # torch.manual_seed(seed)

    c = np.ones(K, dtype=int) * (n // K)

    nmis = []

    ps = np.arange(0, 1.1, 0.1)
    ps = [0.0]
    for p in ps:
        # tqdm.write("p={}".format(p))

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
        # dataset = "fb100-penn94"
        # dataset = "genius"
        # dataset = "citeseer"
        dataset = args.dataset
        K = 2
        tqdm.write("dataset={} {}".format(dataset, K))
        nmis_r = []
        seeds = np.arange(0, args.nexp, dtype=int)
        for seed in tqdm(seeds, total=args.nexp):
            np.random.seed(seed)
            torch.manual_seed(seed)

            setproctitle.setproctitle("{}-{}-{}".format(args.model, args.dataset[:2], seed))

            tqdm.write('Generating graphs ...')
            # dataset = "cora"
            # dcsbm = DCSBM()
            # adj, features, labels = dcsbm.generate(c, w, theta, lamb, random_state=seed, save=False)
            adj, features, labels = load_assortative(dataset=dataset)
            # adj, features, labels = load_syn_h2gcn(dataset=dataset, h=p, r=r)
            # adj, features, labels = load_disassortative(dataset=dataset)
            features = preprocess_features(features)
            adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
            adj.eliminate_zeros()
            n, m, d = adj.shape[0], adj.sum(), features.shape[1]
            tqdm.write("[nodes,edges,features]=[{}, {}, {}]".format(n, m, d))
            adj_norm = preprocess_graph(adj, renorm=True)
            # adj_1st = adj + sp.eye(n)
            adj_1st = (adj + sp.eye(n)).toarray()
            # adj_1st = sparse_mx_to_torch_sparse_tensor(adj + sp.eye(n))
            # plot_superadj(adj, K=min(adj.shape[0], 100), dataset=dataset, sparse=True, labels=labels, vline=True)

            g = dgl.from_scipy(adj_norm)
            g = dgl.to_simple(g)
            g = dgl.remove_self_loop(g)
            g = dgl.to_bidirected(g)
            g = dgl.add_self_loop(g)

            # ids = np.arange(0, n)
            # zero_node_idx = ids[(labels == 0)]
            # one_node_idx = ids[(labels == 1)]
            # tqdm.write(zero_node_idx.shape, one_node_idx.shape)
            # tqdm.write(zero_node_idx[:5], one_node_idx[:5])
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
            # tqdm.write(adj.shape)
            # adj = normalize(adj)
            # s = np.matmul(adj, np.transpose(adj))
            # s = 



            # exit(0)



            # tqdm.write("Heuristic ...")
            # heu = Heuristic()
            # # adj_ = heu.detect3(adj_norm)
            # saved_new_adj = True
            # if not saved_new_adj:
            #     saved_sim = True
            #     if not saved_sim:
            #         adj_ = heu.detect3(adj_1st)
            #         np.savez("outputs/sim_{}.npz".format(dataset), row=adj_.row, col=adj_.col, data=adj_.data)
            #     else:
            #         data_ = np.load("outputs/sim_{}.npz".format(dataset))
            #         row, col, data = data_["row"], data_["col"], data_["data"]
            #         adj_ = sp.coo_matrix((data, (row, col)), shape=(n,n))
            #         adj_.eliminate_zeros()
            #     # # NOTE: 为可视化重排节点
            #     # idx = np.argsort(labels, 0)
            #     # adj_ = adj_[idx, :][:, idx]
            #     # labels = labels[idx]
            #     # features = sp.csr_matrix(features.toarray()[idx, :])
            #     # plot_superadj(adj_.tocsr(), K=min(n, 100), dataset="heuristic_1", sparse=True, labels=labels, vline=True)
            #     thres = heu.set_threshold_nano_sparse(adj_, K=K)
            #     adj_ = heu.normalize_adj(adj_, thres)
            #     tqdm.write('thres={}'.format(thres))
            #     plot_superadj(adj_, K=min(n, 100), dataset="heuristic", sparse=True, labels=labels, vline=True)
            #     np.savez("outputs/newadj_{}.npz".format(dataset), row=adj_.row, col=adj_.col, data=adj_.data)
            # else:
            #     data_ = np.load("outputs/newadj_{}.npz".format(dataset))
            #     row, col, data = data_["row"], data_["col"], data_["data"]
            #     adj_ = sp.coo_matrix((data, (row, col)), shape=(n,n))
            #     adj_.eliminate_zeros()
            #     # plot_superadj(adj_, K=min(n, 100), dataset="heuristic", sparse=True, labels=labels, vline=True)
            # adj = sp.coo_matrix(adj_)
            # adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
            # adj.eliminate_zeros()
            # adj_norm = preprocess_graph(adj, renorm=True)
            # adj_1st = (adj + sp.eye(n)).toarray()


            tqdm.write('Preparing the model ...')
            dims = [features.shape[1]] + [hidden_dim]*(gnn_layers-1) + [emb_dim]
            tqdm.write("dims={}".format(dims))

            if args.model == "GAE":
                model = GAE(gnn_layers, dims, dropout=0.)
            elif args.model == "VGAE":
                model = VGAEModel(dims[0], dims[1], dims[2])
            else:
                raise NotImplementedError()
            # model = DGI(dims[0], dims[1], activation=F.relu, bn=False)
            # model = MVGRL(dims[0], dims[1])
            model = model.to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd) 
            # adj = torch.FloatTensor(adj.toarray()).to(device)
            # adj_norm = sparse_mx_to_torch_sparse_tensor(adj_norm).to(device)
            # adj_ = torch.FloatTensor(adj_.toarray()).to(device)
            adj_label = torch.FloatTensor(adj_1st).to(device)
            features = torch.FloatTensor(features.toarray()).to(device)
            # features = torch.FloatTensor(features.toarray()[np.newaxis]).to(device)
            g = g.to(device)

            pos_weight = torch.tensor(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
            norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)

            for epoch in tqdm(range(epochs), total=epochs):
                model.train()
                adj_rec = model(g, features)
                if args.model == "GAE":
                    loss = norm * model.loss(adj_label, adj_rec, pos_weight=pos_weight) # GAE
                elif args.model == "VGAE":
                    loss = model.loss(adj_label, adj_rec, pos_weight=pos_weight, norm=norm) # VGAE
                else:
                    raise NotImplementedError()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                model.eval()
                # with torch.no_grad():
                #     # emb, _ = model.encoder(features, adj_norm, True, None)
                #     emb, _ = model.embed(features, adj, diff, False, None)
                #     emb = emb[0].cpu().numpy()
                #     cluster = MiniBatchKMeans(n_clusters=K, random_state=0, batch_size=1024, verbose=0)
                #     # cluster = SpectralClustering(n_clusters=K, random_state=0)
                #     preds = cluster.fit_predict(emb)
                #     nmi = NMI(labels, preds)

                #     tqdm.write('epoch {}/{} loss={:.5f} nmi={:.3f}'.format(epoch+1, epochs, loss.item(), nmi))
                if epoch % 10 == 0:
                    tqdm.write("EPOCH {}/{} loss={:.5f}".format(epoch+1, epochs, loss.item()))

            tqdm.write("End of training.")
            tqdm.write('Clustering ...')
            model.eval()
            with torch.no_grad():
                ##### MVGRL start ######
                # features = torch.FloatTensor(features[np.newaxis])
                # adj = torch.FloatTensor(adj[np.newaxis])
                # diff = torch.FloatTensor(diff[np.newaxis])
                # features = features.cuda()
                # adj = adj.cuda()
                # diff = diff.cuda()
                ##### MVGRL end ######
                emb = model.encoder(g, features).cpu().detach().numpy()
                # emb, _ = model.encoder(features, adj_norm, True, None)
                # emb, _ = model.encoder(features, adj, diff, False, None)
                # emb = emb[0].cpu().numpy()
                os.makedirs("outputs", exist_ok=True)
                np.savez("outputs/{}_{}_emb_{}.npz".format(args.model, dataset, seed), emb=emb, labels=labels)
                # emb = normalize(emb)
                # f_adj = np.matmul(emb, np.transpose(emb))
                # tqdm.write("mean={}, max={}, min={}".format(f_adj.mean(), f_adj.max(), f_adj.min()))
                # cluster = KMeans(n_clusters=K, random_state=10)
                # cluster = MiniBatchKMeans(n_clusters=K, random_state=0, batch_size=1024, verbose=0)
                # # cluster = SpectralClustering(n_clusters=K, random_state=0)
                # preds = cluster.fit_predict(emb)
                # nmi = NMI(labels, preds)
                # nmis_r.append(nmi)

                # tqdm.write('Plotting ...')
                # tsne_z = TSNE(n_components=2, init="random").fit_transform(emb)
                # plt.figure(figsize=(10,10))
                # plt.scatter(tsne_z[:,0], tsne_z[:,1], c=labels)
                # import os
                # os.makedirs("pics", exist_ok=True)
                # plt.savefig("pics/cluster.png")


                os.makedirs("models", exist_ok=True)
                torch.save(model.state_dict(), "models/{}_{}_{}.statedict".format(args.model, dataset, seed))

        # nmis.append(np.mean(nmis_r))
        # tqdm.write("nmi={:.3f}".format(np.mean(nmis_r)))


    # os.makedirs("results", exist_ok=True)
    # with open("results/results.txt", 'a+') as f:
    #     for nmi in nmis:
    #         f.write('{} '.format(nmi))
    #     f.write("\n\n")





