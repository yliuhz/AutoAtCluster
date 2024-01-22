
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch import optim
from tqdm import tqdm
import time
from sklearn.preprocessing import normalize, MinMaxScaler, StandardScaler
from sklearn.cluster import MiniBatchKMeans
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)

from utils import *
from vis import plot_superadj
from load import load_assortative, load_cora_full_diff_cls
import matplotlib.pyplot as plt

dataset2K = {
    "cora": 7
}

import setproctitle
import os
import scipy.sparse as sp

import random
from memory_profiler import profile
import tracemalloc

class BinaryClassification(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_layers=2, dropout=0.2):
        super(BinaryClassification, self).__init__()
        # Number of input features is 12.
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(in_dim, hidden_dim))
        for i in range(n_layers-2): 
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.layer_out = nn.Linear(hidden_dim, 1) 
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)
        self.batchnorm1 = nn.BatchNorm1d(hidden_dim)
        self.batchnorm2 = nn.BatchNorm1d(hidden_dim)

        self.mlp = nn.Linear(in_dim, 1)

    #     self.init_weights()

    # def init_weights(self):
    #     nn.init.xavier_normal_(self.layer_1.weight, gain=1.414)
    #     nn.init.xavier_normal_(self.layer_2.weight, gain=1.414)
    #     nn.init.xavier_normal_(self.layer_out.weight, gain=1.414)

    #     # nn.init.xavier_normal_(self.batchnorm1.weight, gain=1.414)
    #     # nn.init.xavier_normal_(self.batchnorm2.weight, gain=1.414)
    
        
    def forward(self, inputs):
        x = inputs
        for layer in self.layers:
            x = layer(x)
            x = self.relu(x)
        # x = self.layer_1(inputs)
        # x = self.relu(x)
        # x = self.batchnorm1(x)
        # x = self.layer_2(x)
        # x = self.relu(x)
        # x = self.batchnorm2(x)
        x = self.dropout(x)
        x = self.layer_out(x)

        # x = self.mlp(x)
        
        return x.reshape(-1)


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

import argparse
def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='ogbn-products', help='type of dataset.')
    parser.add_argument('--nexp', type=int, default=1, help="Number of repeated experiments")
    parser.add_argument("--save_model", action='store_true', help='Whether to store the link model')
    parser.add_argument("--model", type=str, default="GGD_product")
    parser.add_argument("--pos_k_times_m", type=int, default=10, help="pos=k*m/n")
    parser.add_argument("--neg", type=float, default=0.9, help="ratio of left indices after knn")
    parser.add_argument("--gnnlayers", type=int, default=2)
    

    ## Grid search
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--scaler", type=str, default="minmax")
    parser.add_argument("--mlp_layers", type=int, default=2)

    parser.add_argument("--nclass", type=int, default=5)

    return parser

def sampling(adj, rate=0.5, random_state=None):
    if rate >= 1.0:
        return adj

    n = adj.shape[0]
    adj = adj.toarray()
    
    ret = np.zeros((n,n))
    
    for i in range(n):
        row_idx = adj[i].nonzero()[0]
        arr = np.random.RandomState(seed=random_state).choice(row_idx, int(rate*row_idx.shape[0]), replace=False)
        ret[i][arr] = 1
    
    return sp.coo_matrix(ret)

def loss_function(adj_preds, adj_labels):
    cost = 0.
    cost += F.binary_cross_entropy_with_logits(adj_preds, adj_labels)
    
    return cost


# @profile
def main():
    parser = make_parser()
    
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
        "GGD": "/data/liuyue/New/SBM/mySBM/emb_models/GGD/manual_version/outputs",
        "GGD_product": "/data/liuyue/New/SBM/mySBM/emb_models/GGD/GGD_ogbn_product_1epoch/outputs",
    }

    GRACE_datasets = {
        "cora": "Cora",
        "citeseer": "CiteSeer", 
        "wiki": "Wiki", 
        "pubmed": "PubMed",
        "amazon-photo": "amazon-photo",
        "amazon-computers": "amazon-computers",
        "cora-full": "cora-full",
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
        random.seed(seed)
        torch.cuda.manual_seed(seed)

        setproctitle.setproctitle("LKE-{}-{}-{}".format(dataset[:2], args.model, seed))

        if args.model == "GRACE":
            dataset_ = GRACE_datasets[dataset]
        else:
            dataset_ = dataset

        adj, features, true_labels = load_assortative(dataset)
        # adj, features, true_labels, mask = load_cora_full_diff_cls(args.nclass, seed)
        # adj, _, _ = load_syn_subgraph(dataset=dataset)

        # plot_superadj(adj, K=100, sparse=True, labels=true_labels, dataset="cora-full_{:.1f}_{:d}".format(args.nclass, seed), vline=True)

        # exit(0)
        print(adj.shape[0], adj.sum())

        emb_path = os.path.join(emb_paths[args.model], "{}_{}_emb_{}.npz".format(args.model.split("_")[0], dataset_, seed))
        data = np.load(emb_path)
        # adj, _, true_labels = load_assortative(dataset=dataset)
        # adj, _, _ = load_syn_subgraph(dataset=dataset)

        # emb, true_labels = data["emb"], data["labels"]
        emb = data["emb"]

        # import matplotlib.pyplot as plt
        # from sklearn.manifold import TSNE
        # tsne_z = TSNE(n_components=2, random_state=seed).fit_transform(sm_fea_s)
        # plt.figure(figsize=(10,10))
        # plt.scatter(tsne_z[:, 0], tsne_z[:, 1], c=true_labels)
        # plt.savefig("pics/tsne_{}_{}.png".format(dataset, seed))
        
        if args.scaler == "minmax":
            scaler = MinMaxScaler()
        elif args.scaler == "standard":
            scaler = StandardScaler()
        sm_fea_s = scaler.fit_transform(emb)
        emb = torch.FloatTensor(sm_fea_s)

        model_config = {
            "in_dim": emb.shape[-1],
            "hidden_dim": 500,
            "out_epochs": args.num_epochs//10, # 800 
            "in_epochs": 10,
            "lr": args.lr,
            "device": "cuda:0",
            # "batch_size": min(emb.shape[0], 4096), # 1024
            "sample_size": 256, # 1024
            "print_epoch": 10,
            "patience": 10,
            "upd": 50,
            "m": adj.sum(),
        }

        inx = torch.FloatTensor(emb).to(model_config["device"])

        model = BinaryClassification(model_config["in_dim"], model_config["hidden_dim"], n_layers=args.mlp_layers).to(model_config["device"])
        print(model)

        print("Loading model")
        model.load_state_dict(torch.load("saved_models/GGD_ogbn_products_best.pkl"))
        print("Loaded model success")


        with torch.no_grad():
            model.eval()

            # mu = model(inx)
            # hidden_emb = mu.cpu().data.numpy()
            # import os
            # os.makedirs("outputs", exist_ok=True)
            # np.savez("outputs/AGE_{}_emb_{}.npz".format(args.dataset, seed), emb=hidden_emb, labels=true_labels)       
            row, col, data = [], [], []
            # arr = np.arange(emb.shape[0], dtype=int)
            # M_edges = int(np.log(emb.shape[0]))
            # M_edges = emb.shape[0]
            # print("M_edges={}".format(M_edges))

            # NOTE: N*LOGN edges
            batch_size = 1024
            for i in tqdm(range(emb.shape[0]), total=emb.shape[0]):
                st = 0
                while st < emb.shape[0]:
                    ed = min(st+batch_size, emb.shape[0])
                    arr = np.arange(st, ed, 1, dtype=int)

                    inputs = (inx[i] * inx[st:ed]).to(model_config["device"])

                    # zmax = inputs.max(dim=1, keepdim=True)[0]
                    # zmin = inputs.min(dim=1, keepdim=True)[0]
                    # inputs = (inputs - zmin) / (zmax - zmin)
                    # inputs = F.normalize(inputs)

                    preds = torch.sigmoid(model(inputs))

                    # preds[preds > 0.5] = 1.
                    # preds[preds <= 0.5] = 0.
                    neib_ids = arr[preds.cpu() > 0.5]
                    # if neib_ids.shape[0] > M_edges:
                    #     neib_ids = np.random.permutation(neib_ids)
                    #     neib_ids = neib_ids[:M_edges]

                    # data += (preds.cpu().tolist())
                    # row += ([i] * emb.shape[0])
                    # col += ([x for x in range(emb.shape[0])])

                    data += [1] * neib_ids.shape[0]
                    row += [i] * neib_ids.shape[0]
                    col += neib_ids.tolist()

                    st += batch_size

            adj = sp.coo_matrix((data, (row, col)), shape=(emb.shape[0], emb.shape[0]))
            adj.eliminate_zeros()

            os.makedirs("link_adj_knn/{}".format(args.model), exist_ok=True)
            np.savez("link_adj_knn/{}/{}_{}.npz".format(args.model, args.dataset, seed), data=adj.data, row=adj.row, col=adj.col)

            # if args.save_model:
            #     torch.save(model.state_dict(), "saved_models/{}-{}-{}.statedict".format(args.model, args.dataset, seed))


            # m2 = adj.sum()
            # if m2 > 10*model_config["m"]:
            #     adj_s = sampling(adj, rate=10*model_config["m"]/adj.sum(), random_state=seed)
            # else:
            #     adj_s = adj
            # # adj_s = sampling(adj, rate=sampling_rate)
            # adj = adj_s

            # tqdm.write('n={}, m={}'.format(adj.shape[0], adj.sum()))   
            # os.makedirs("gs", exist_ok=True) 
            # np.savez("gs/{}_{}.npz".format(dataset, seed), data=adj.data, col=adj.col, row=adj.row)

            plot_superadj(adj, K=min(100, adj.shape[0]), sparse=True, labels=true_labels, vline=True, dataset="link") 

            st1 = time.process_time()
            preds = louvain_cluster(adj, true_labels, random_state=seed)
            ed1 = time.process_time()
            print(f"Time of Louvain: {ed1-st1:.2f} sec")

            labels = true_labels

            from sklearn.metrics import normalized_mutual_info_score as NMI, adjusted_mutual_info_score as AMI, adjusted_rand_score as ARI

            nmi = NMI(labels, preds)
            ami = AMI(labels, preds)
            ari = ARI(labels, preds)
            nclass = np.unique(preds).shape[0]

            print(nmi, ami, ari, nclass)

if __name__ == "__main__":
    main()