
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch import optim
from tqdm import tqdm
import time
from sklearn.preprocessing import normalize
from sklearn.cluster import MiniBatchKMeans
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)

from layers import *
from utils import *

dataset2K = {
    "cora": 7
}

import setproctitle
import os
import scipy.sparse as sp

from vis import plot_superadj

def louvain_cluster(adj, labels, random_state=None):
    from community import community_louvain
    import networkx as nx
    from sklearn.metrics import normalized_mutual_info_score as NMI, adjusted_mutual_info_score as AMI

    graph = nx.from_scipy_sparse_matrix(adj)
    partition = community_louvain.best_partition(graph, random_state=random_state)
    preds = list(partition.values())

    return preds

class LinTrans(nn.Module):
    def __init__(self, layers, dims):
        super(LinTrans, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(layers):
            self.layers.append(nn.Linear(dims[i], dims[i+1]))
            # self.layers.append(GCN(dims[i], dims[i+1], 'prelu'))
        self.dcs = SampleDecoder(act=lambda x: x)

    def scale(self, z):
        
        zmax = z.max(dim=1, keepdim=True)[0]
        zmin = z.min(dim=1, keepdim=True)[0]
        z_std = (z - zmin) / (zmax - zmin)
        z_scaled = z_std
    
        return z_scaled

    def forward(self, x):
        out = x
        for layer in self.layers:
            out = layer(out)
        out = self.scale(out)
        out = F.normalize(out)
        return out

class SampleDecoder(nn.Module):
    def __init__(self, act=torch.sigmoid):
        super(SampleDecoder, self).__init__()
        self.act = act

    def forward(self, zx, zy):
        sim = (zx * zy).sum(1)
        sim = self.act(sim) # act(x)=x
    
        return sim

def loss_function(adj_preds, adj_labels):
    cost = 0.
    cost += F.binary_cross_entropy_with_logits(adj_preds, adj_labels)
    
    return cost

# def update_similarity(z, upper_threshold, lower_treshold, pos_num, neg_num):
def update_similarity(z, pos_num, neg_num):
    f_adj = np.matmul(z, np.transpose(z)) # sim
    cosine = f_adj
    cosine = cosine.reshape([-1,])
    # pos_num = round(upper_threshold * len(cosine))
    # neg_num = round((1-lower_treshold) * len(cosine))
    
    pos_inds = np.argpartition(-cosine, pos_num)[:pos_num]
    neg_inds = np.argpartition(cosine, neg_num)[:neg_num]
    
    # return np.array(pos_inds), np.array(neg_inds)
    return torch.LongTensor(pos_inds), torch.LongTensor(neg_inds)

def update_threshold(upper_threshold, lower_treshold, up_eta, low_eta):
    upth = upper_threshold + up_eta
    lowth = lower_treshold + low_eta
    return upth, lowth

import argparse
def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cora', help='type of dataset.')
    parser.add_argument('--nexp', type=int, default=10, help="Number of repeated experiments")
    parser.add_argument("--save_model", action='store_true', help='Whether to store the link model')
    return parser



if __name__ == "__main__":
    parser = make_parser()
    parser.add_argument("--model", type=str, default="GAE")
    parser.add_argument("--pos", type=float, default=0.01)
    parser.add_argument("--neg", type=float, default=0.9)
    parser.add_argument("--gnnlayers", type=int, default=2)
    args = parser.parse_args()
    print(args)

    dataset = args.dataset
    print("dataset={}".format(dataset))
    results_lo, results_km = [], []

    emb_paths = {
        "GAE": "/data/liuyue/New/SBM/mySBM/emb_models/GAE/outputs", 
        "VGAE": "/data/liuyue/New/SBM/mySBM/emb_models/GAE/outputs",
        "ARGA": "/data/liuyue/New/SBM/mySBM/emb_models/ARGA/ARGA/arga/outputs",
        "ARVGA": "/data/liuyue/New/SBM/mySBM/emb_models/ARGA/ARGA/arga/outputs",
        "AGE": "/data/liuyue/New/SBM/mySBM/emb_models/AGE/outputs",
        "DGI": "/data/liuyue/New/SBM/mySBM/emb_models/DGI/outputs",
        "MVGRL": "/data/liuyue/New/SBM/mySBM/emb_models/MVGRL/outputs",
        "GRACE": "/data/liuyue/New/SBM/mySBM/emb_models/GRACE/outputs",
        "GGD": "/data/liuyue/New/SBM/mySBM/emb_models/GGD/manual_version/outputs"
    }

    GRACE_datasets = {
        "cora": "Cora",
        "citeseer": "CiteSeer", 
        "wiki": "Wiki", 
        "pubmed": "PubMed",
        "amazon-photo": "amazon-photo",
        "amazon-computers": "amazon-computers"
    }

    nclasses = {
        "cora": 7,
        "citeseer": 6,
        "pubmed": 3,
    }

    seeds = np.arange(0, args.nexp, dtype=int) # seed = 0
    # seeds = [11]
    # seeds = [7]
    for seed in tqdm(seeds, total=len(seeds)):
        np.random.seed(seed)
        torch.manual_seed(seed)

        setproctitle.setproctitle("AGEvs-{}-{}".format(dataset[:2], seed))

        if args.model == "GRACE":
            dataset_ = GRACE_datasets[dataset]
        else:
            dataset_ = dataset

        adj, features, true_labels, _, _, _ = load_data(dataset)
        # adj, _, _ = load_syn_subgraph(dataset=dataset)

        # emb, true_labels = data["emb"], data["labels"]
        # emb = data["emb"]
        # zmax = emb.max(axis=1, keepdims=True)
        # zmin = emb.min(axis=1, keepdims=True)
        # emb = (emb - zmin) / (zmax - zmin)

        # print('Emb.shape={}'.format(emb.shape))

        adj_norm_s = preprocess_graph(adj, args.gnnlayers, norm='sym', renorm=True)
        sm_fea_s = sp.csr_matrix(features).toarray()
        
        print('Laplacian Smoothing...')
        # best_db = -999
        # best_layers = 0
        for idx, a in enumerate(adj_norm_s):
            sm_fea_s = a.dot(sm_fea_s)
        emb = torch.FloatTensor(sm_fea_s)

        # print("Load Embedding")
        # emb_path = os.path.join(emb_paths[args.model], "{}_{}_emb_{}.npz".format(args.model, dataset_, seed))
        # data = np.load(emb_path)
        # emb = data["emb"]


        model_config = {
            "in_dim": emb.shape[-1],
            "hidden_dim": 500,
            "epochs": 400, # 800 
            "lr": 1e-3,
            "device": "cuda:0",
            # "batch_size": min(emb.shape[0], 4096), # 1024
            "sample_size": 10000, # 1024
            "print_epoch": 10,
            "patience": 10,
            "upd": 50,
            "m": adj.sum(),

        }
        model_config["sample_size"] = min(model_config["sample_size"], model_config["m"])

        inx = torch.FloatTensor(emb).to(model_config["device"])
        # inx = F.normalize(inx)

        model = LinTrans(1, [model_config["in_dim"], model_config["hidden_dim"]]).to(model_config["device"])
        optimizer = optim.Adam(model.parameters(), lr=model_config["lr"], eps=1e-3)
        # criterion = nn.BCEWithLogitsLoss()
        criterion = loss_function
        print(model)

        best_loss = 100
        early = 0
        # sim = normalize(emb)
        # sim = np.matmul(sim, np.transpose(sim))
        # sim_argsort = np.argsort(sim)

        # pos_num = model_config["sample_size"]
        data = np.load("/data/liuyue/New/SBM/mySBM/params_/{}-{}.npz".format(args.model, dataset))
        # pos_r, neg_r = args.pos, 1-args.neg
        pos_r, neg_r = data["pos"], data["neg"]
        # pos_num = model_config["m"]
        # pos_num = 80665
        pos_num = round(emb.shape[0] * emb.shape[0] * pos_r)
        # neg_num = pos_num * 5 # pos_num * 5
        neg_num = round(emb.shape[0] * emb.shape[0] * neg_r)
        pos_inds, neg_inds = update_similarity(normalize(emb), pos_num, neg_num)

        # pos_inds, neg_inds, test_inds = update_similarity(normalize(emb), pos_num, neg_num, test=True)

        # test_inds = test_inds.to(model_config["device"])
        # test_x = torch.index_select(inx, 0, test_inds // emb.shape[0])
        # test_y = torch.index_select(inx, 0, test_inds % emb.shape[0])
        # test_inputs = (test_x * test_y).to(model_config["device"])

        # # num_03_07 = test_inds.shape[0]
        # print(pos_inds.shape, neg_inds.shape, test_inds.shape, emb.shape[0] ** 2)

        # nn = np.arange(emb.shape[0], dtype=int)

        # Y = true_labels
        # Y_sim = np.zeros((adj.shape[0], adj.shape[0]), dtype=int)
        # for i in range(adj.shape[0]):
        #     Y_sim[i] = (Y[i] == Y)
            
        # plt.figure(figsize=(10,10))
        # nodeid = 100
        # sim_idx = np.argsort(-sim[nodeid])
        # sim_ = sim[nodeid][sim_idx]
        # plt.scatter([x for x in range(sim.shape[0])], sim_)
        # # plt.colorbar()
        # plt.savefig("pics/sim_node{}_{}.png".format(nodeid, dataset))

        # plt.figure(figsize=(10,10))
        # tsne_z = TSNE(n_components=2).fit_transform(emb)
        # plt.scatter(tsne_z[:, 0], tsne_z[:, 1], c=true_labels)
        # plt.colorbar()
        # plt.savefig("pics/tsne_{}_{}.png".format(args.model, dataset))

        # plt.figure(figsize=(10,10))
        # plt.scatter(tsne_z[:, 0], tsne_z[:, 1], c=sim[nodeid])
        # plt.colorbar()
        # plt.savefig("pics/tsne_node{}_{}.png".format(nodeid, dataset))
        # exit(0)


        # adj = np.zeros((emb.shape[0], emb.shape[0]))
        # xinds = pos_inds // emb.shape[0]
        # yinds = pos_inds % emb.shape[0]
        # for x, y in zip(xinds, yinds):
        #     adj[x][y] = 1.
        # plot_superadj(adj, K=100, sparse=False, labels=true_labels, vline=True, dataset="pos")
        # adj = np.zeros((emb.shape[0], emb.shape[0]))
        # xinds = neg_inds // emb.shape[0]
        # yinds = neg_inds % emb.shape[0]
        # for x, y in zip(xinds, yinds):
        #     adj[x][y] = 1.
        # plot_superadj(adj, K=100, sparse=False, labels=true_labels, vline=True, dataset="neg")
        # adj = np.zeros((emb.shape[0], emb.shape[0]))
        # xinds = test_inds // emb.shape[0]
        # yinds = test_inds % emb.shape[0]
        # for x, y in zip(xinds, yinds):
        #     adj[x][y] = 1.
        # plot_superadj(adj, K=100, sparse=False, labels=true_labels, vline=True, dataset="test_inds")
        # exit(0)


        # sim = emb.copy()
        # zmax = sim.max(axis=1, keepdims=True)
        # zmin = sim.min(axis=1, keepdims=True)
        # sim = (sim - zmin) / (zmax - zmin)
        # sim = normalize(sim)
        # sim = np.matmul(sim, np.transpose(sim)).reshape(-1)
        # pos_t = 0.95
        # neg_t = 0.45
        # pos_inds, neg_inds = update(sim, pos_t, neg_t)
        model_config["sample_size"] = min(model_config["sample_size"], pos_inds.shape[0])


        for epoch in tqdm(range(model_config["epochs"]), total=model_config["epochs"]):
            model.train()

            # idx = np.random.choice(np.arange(emb.shape[0]), model_config["batch_size"])
            # b_emb = emb[idx, :]
            # b_emb = emb

            # b_sim = b_emb.copy()
            # # b_sim = torch.FloatTensor(b_sim)
            # # b_sim = torch.cdist(b_sim, b_sim).numpy().reshape(-1)
            # # b_sim = sim[idx, :][:, idx].reshape(-1)
            # b_sim = normalize(b_sim)
            # b_sim = np.matmul(b_sim, np.transpose(b_sim)).reshape(-1)
            # # b_sim = sim[idx, :][:, idx].reshape(-1)

            # # print("min={}".format(b_sim.min()))

            # pos_num = model_config["sample_size"]
            # neg_num = pos_num * 10 # pos_num * 5


            # # sim_argsort = np.argsort(b_sim)
            # pos_inds = torch.LongTensor(np.argpartition(-b_sim, pos_num)[:pos_num])
            # neg_inds = torch.LongTensor(np.argpartition(b_sim, neg_num)[:neg_num])
            # pos_inds = torch.LongTensor(sim_argsort[-pos_num:])
            # neg_inds = torch.LongTensor(sim_argsort[:neg_num])
            # sampled_inds = torch.cat([pos_inds, neg_inds]).to(model_config["device"])

            st, ed = 0, model_config["sample_size"]
            length = len(pos_inds)
            while ed <= length:
                sampled_neg = torch.LongTensor(np.random.choice(neg_inds.numpy(), size=ed-st))
                sampled_inds = torch.cat((pos_inds[st:ed], sampled_neg), 0).to(model_config["device"])
                xinds = sampled_inds // emb.shape[0]
                yinds = sampled_inds % emb.shape[0]

                # x = b_emb[pos_inds]
                # y = b_emb[neg_inds]

                x = torch.index_select(inx, 0, xinds)
                y = torch.index_select(inx, 0, yinds)

                zx = model(x)
                zy = model(y)
                labels = torch.cat((torch.ones(ed-st), torch.zeros(ed-st))).to(model_config["device"])
                preds = model.dcs(zx, zy)

                # x = torch.FloatTensor(x).to(model_config["device"])
                # y = torch.FloatTensor(y).to(model_config["device"])

                # labels = torch.cat([torch.ones(ed - st), torch.zeros(ed - st)]).to(model_config["device"])
                # inputs = (x * y).to(model_config["device"])
                # zmax = inputs.max(dim=1, keepdim=True)[0]
                # zmin = inputs.min(dim=1, keepdim=True)[0]
                # inputs = (inputs - zmin) / (zmax - zmin)
                # inputs = F.normalize(inputs)

                # preds = model(inputs)
                
                optimizer.zero_grad()
                loss = criterion(preds, labels)
                loss.backward()
                optimizer.step()

                st = ed
                if ed < length and ed + model_config["sample_size"] >= length:
                    ed += length - ed
                else:
                    ed += model_config["sample_size"]
    
            if epoch % model_config["print_epoch"] == 0:
                tqdm.write("EPOCH {:d}/{:d} Loss:{:.4f}".format(epoch+1, model_config["epochs"], loss.item()))

            if loss > best_loss:
                early += 1
            else:
                best_loss = loss
                early = 0
            
            if early > model_config["patience"]:
                tqdm.write("Early stop at {} epoch".format(epoch))
                break

            # if epoch % model_config["upd"] == 0:
            #     # model.eval()
            #     # with torch.no_grad():
            #     #     test_preds = model(test_inputs).cpu()
            #     #     test_preds = torch.sigmoid(test_preds)

            #     #     plt.figure()
            #     #     plt.hist(test_preds.numpy())
            #     #     plt.savefig("pics/hist_test_preds.png")


            #     #     new_num_03_07 = ((test_preds >= 0.1) * (test_preds <= 0.9)).sum()
            #     #     if new_num_03_07 > num_03_07:
            #     #         break
            #     #     num_03_07 = new_num_03_07
            #     #     tqdm.write("num_03_07={}".format(num_03_07))

            #     # pos_num += model_config["m"]
            #     # neg_num += model_config["m"]

            #     lamb = 0.05
            #     pos_t -= lamb
            #     neg_t += 0.1 * lamb
            #     pos_inds, neg_inds = update(sim, pos_t, neg_t)
            #     model_config["sample_size"] = min(model_config["sample_size"], pos_inds.shape[0])

            # pos_t = pos_t -  (0.95 - 0.5) / model_config["epochs"]
            # pos_inds, neg_inds = update(sim, pos_t, neg_t)


                # pos_inds, neg_inds = update_similarity(normalize(emb), pos_num, neg_num)


        with torch.no_grad():
            model.eval()

            mu = model(inx)
            hidden_emb = mu.cpu().data.numpy()

            # os.makedirs("OursvsAGE", exist_ok=True)
            # np.savez("OursvsAGE/AGE_{}_{}_{}.npz".format(args.model, args.dataset, seed), emb=hidden_emb)
            
            sim = np.matmul(hidden_emb, np.transpose(hidden_emb))
            sim[sim > 0.5] = 1
            sim[sim <= 0.5] = 0

            import scipy.sparse as sp
            sim = sp.coo_matrix(sim)

            print(sim.shape[0], sim.sum())
            plot_superadj(sim, K=min(100, adj.shape[0]), sparse=True, labels=true_labels, vline=True, dataset="adaptive_link")

            # preds = louvain_cluster(sim, true_labels, random_state=seed)

            # from sklearn.metrics import normalized_mutual_info_score as NMI, adjusted_mutual_info_score as AMI, adjusted_rand_score as ARI

            # nmi = NMI(true_labels, preds)
            # ami = AMI(true_labels, preds)
            # ari = ARI(true_labels, preds)

            # print(nmi, ami, ari)        

            if args.save_model:
                os.makedirs("saved_models", exist_ok=True)
                torch.save(model.state_dict(), "saved_models/AGEv-{}-{}.statedict".format(args.dataset, seed))


    # os.makedirs("results", exist_ok=True)
    # with open("results/results.txt", 'a+') as f:
    #     f.write("\n\n\n")
    #     # for alg, data in results.items():
    #     #     f.write("alg={}, k={}-{}".format(alg, Kmin, Kmax))
    #     #     for k, values in data.items():
    #     #         # f.write("{}-{}\n".format(alg, k))
    #     #         for d in values:
    #     #             f.write("{} ".format(d))
    #     #         f.write("\n")

    #     f.write("alg=louvain\n")
    #     for values in results_lo:
    #         f.write("{} ".format(values))
    #     f.write("\n")

    #     f.write("alg=KMeans\n")
    #     for values in results_km:
    #         f.write("{} ".format(values))
    #     f.write("\n")







