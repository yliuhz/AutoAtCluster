
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

from layers import *
from utils import *
from vis import plot_superadj

dataset2K = {
    "cora": 7
}

import setproctitle
import os
import scipy.sparse as sp

import random

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
        
        return x.reshape(-1)

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
    parser.add_argument('--dataset', type=str, default='cora', help='type of dataset.')
    parser.add_argument('--nexp', type=int, default=10, help="Number of repeated experiments")
    parser.add_argument("--save_model", action='store_true', help='Whether to store the link model')
    parser.add_argument("--model", type=str, default="GAE")
    parser.add_argument("--pos", type=float, default=0.01)
    parser.add_argument("--neg", type=float, default=0.9)
    parser.add_argument("--gnnlayers", type=int, default=2)
    


    ## Grid search
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--scaler", type=str, default="minmax")
    parser.add_argument("--mlp_layers", type=int, default=2)
    parser.add_argument("--epoch", type=int, default=100)


    return parser

def sampling(adj, rate=0.5, random_state=None):
    n = adj.shape[0]
    adj = adj.toarray()
    
    ret = np.zeros((n,n))
    
    for i in range(n):
        row_idx = adj[i].nonzero()[0]
        arr = np.random.RandomState(seed=random_state).choice(row_idx, int(rate*row_idx.shape[0]))
        ret[i][arr] = 1
    
    return sp.coo_matrix(ret)

def loss_function(adj_preds, adj_labels):
    cost = 0.
    cost += F.binary_cross_entropy_with_logits(adj_preds, adj_labels)
    
    return cost





if __name__ == "__main__":
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
        random.seed(seed)
        torch.cuda.manual_seed(seed)

        setproctitle.setproctitle("Oursvs-{}-{}".format(dataset[:2], seed))

        if args.model == "GRACE":
            dataset_ = GRACE_datasets[dataset]
        else:
            dataset_ = dataset

        adj, features, true_labels, _, _, _ = load_data(dataset)

        adj_norm_s = preprocess_graph(adj, args.gnnlayers, norm='sym', renorm=True)
        sm_fea_s = sp.csr_matrix(features).toarray()
        
        print('Laplacian Smoothing...')
        # best_db = -999
        # best_layers = 0
        for idx, a in enumerate(adj_norm_s):
            sm_fea_s = a.dot(sm_fea_s)
        raw_emb = sm_fea_s.copy()
        
        if args.scaler == "minmax":
            scaler = MinMaxScaler()
        elif args.scaler == "standard":
            scaler = StandardScaler()
        sm_fea_s = scaler.fit_transform(sm_fea_s)
        emb = torch.FloatTensor(sm_fea_s)

        model_config = {
            "in_dim": emb.shape[-1],
            "hidden_dim": 300,
            "epochs": args.num_epochs, # 800 
            "lr": args.lr,
            "device": "cuda:0",
            # "batch_size": min(emb.shape[0], 4096), # 1024
            "sample_size": 10000, # 1024
            "print_epoch": 10,
            "patience": 10,
            "upd": 50,
            "m": adj.sum(),
        }

        inx = torch.FloatTensor(emb).to(model_config["device"])


        # valid_xinds, valid_yinds = adj.nonzero()
        # valid_xinds = torch.LongTensor(valid_xinds).to(model_config["device"])
        # valid_yinds = torch.LongTensor(valid_yinds).to(model_config["device"])
        # valid_x = torch.index_select(inx, 0, valid_xinds).to(model_config["device"])
        # valid_y = torch.index_select(inx, 0, valid_yinds).to(model_config["device"])
        # valid_inputs = (valid_x * valid_y).to(model_config["device"])
        # valid_labels = torch.ones(valid_x.shape[0]).to(model_config["device"])

        model = BinaryClassification(model_config["in_dim"], model_config["hidden_dim"], n_layers=args.mlp_layers).to(model_config["device"])
        # model = LinTrans(1, (model_config["in_dim"], model_config["hidden_dim"])).to(model_config["device"])
        optimizer = optim.Adam(model.parameters(), lr=model_config["lr"], eps=1e-3)
        criterion = nn.BCEWithLogitsLoss()
        # criterion = loss_function
        print(model)

        best_loss = 100
        early = 0
        pos_r, neg_r = args.pos, 1-args.neg
        print(pos_r, neg_r)
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
        bs = min(model_config["sample_size"], len(pos_inds))

        losses = []
        valid_losses = []

        for epoch in tqdm(range(model_config["epochs"]), total=model_config["epochs"]):
            model.train()
            st, ed = 0, bs
            length = len(pos_inds)
            while ed <= length:
                sampled_neg = torch.LongTensor(np.random.choice(neg_inds.numpy(), size=ed-st))
                sampled_inds = torch.cat((pos_inds[st:ed], sampled_neg), 0).to(model_config["device"])
                xinds = sampled_inds // emb.shape[0]
                yinds = sampled_inds % emb.shape[0]
                x = torch.index_select(inx, 0, xinds).to(model_config["device"])
                y = torch.index_select(inx, 0, yinds).to(model_config["device"])
                labels = torch.cat([torch.ones(ed - st), torch.zeros(ed - st)]).to(model_config["device"])
                inputs = (x * y).to(model_config["device"])
                # zmax = inputs.max(dim=1, keepdim=True)[0]
                # zmin = inputs.min(dim=1, keepdim=True)[0]
                # inputs = (inputs - zmin) / (zmax - zmin)
                # inputs = F.normalize(inputs)

                # noise = torch.randn_like(inputs).to(model_config["device"])
                # inputs = inputs + noise

                preds = model(inputs)

                
                
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

                # with torch.no_grad():
                #     model.eval()
                #     valid_preds = model(valid_inputs)
                #     valid_loss = criterion(valid_preds, valid_labels)

                #     losses.append(loss.detach().cpu().item())
                #     valid_losses.append(valid_loss.detach().cpu().item())

                # tqdm.write("EPOCH {:d}/{:d} Loss:{:.4f} Valid Loss:{:.4f}".format(epoch+1, model_config["epochs"], loss.item(), valid_loss.item()))





            if loss > best_loss:
                early += 1
            else:
                best_loss = loss
                early = 0
            
            # if early > model_config["patience"]:
            #     tqdm.write("Early stop at {} epoch".format(epoch))
            #     break

        # plt.figure()
        # plt.plot([x for x in range(len(losses))], losses, "r", label="loss")
        # plt.plot([x for x in range(len(valid_losses))], valid_losses, "b", label="v_loss")
        # plt.legend()
        # plt.savefig("pics/loss.png")


        with torch.no_grad():
            model.eval()

            # mu = model(inx)
            # hidden_emb = mu.cpu().data.numpy()
            # import os
            # os.makedirs("outputs", exist_ok=True)
            # np.savez("outputs/AGE_{}_emb_{}.npz".format(args.dataset, seed), emb=hidden_emb, labels=true_labels)       
            row, col, data = [], [], []
            arr = np.arange(emb.shape[0], dtype=int)
            M_edges = int(np.log(emb.shape[0]))
            M_edges = emb.shape[0]
            print("M_edges={}".format(M_edges))

            # NOTE: N*LOGN edges
            for i in tqdm(range(emb.shape[0]), total=emb.shape[0]):
                inputs = (inx[i] * inx).to(model_config["device"])

                # zmax = inputs.max(dim=1, keepdim=True)[0]
                # zmin = inputs.min(dim=1, keepdim=True)[0]
                # inputs = (inputs - zmin) / (zmax - zmin)
                # inputs = F.normalize(inputs)

                preds = torch.sigmoid(model(inputs))

                # preds[preds > 0.5] = 1.
                # preds[preds <= 0.5] = 0.
                neib_ids = arr[preds.cpu() > 0.5]
                if neib_ids.shape[0] > M_edges:
                    neib_ids = np.random.permutation(neib_ids)
                    neib_ids = neib_ids[:M_edges]

                # data += (preds.cpu().tolist())
                # row += ([i] * emb.shape[0])
                # col += ([x for x in range(emb.shape[0])])

                data += [1] * neib_ids.shape[0]
                row += [i] * neib_ids.shape[0]
                col += neib_ids.tolist()

            adj = sp.coo_matrix((data, (row, col)), shape=(emb.shape[0], emb.shape[0]))
            adj.eliminate_zeros()

            # m2 = adj.sum()
            # if m2 > 10*model_config["m"]:
            #     adj_s = sampling(adj, rate=10*model_config["m"]/adj.sum(), random_state=seed)
            # else:
            #     adj_s = adj
            # # adj_s = sampling(adj, rate=sampling_rate)
            # adj = adj_s

            tqdm.write('n={}, m={}'.format(adj.shape[0], adj.sum()))   
            os.makedirs("OursvsAGE/Ours", exist_ok=True) 
            np.savez("OursvsAGE/Ours/{}_{}.npz".format(dataset, seed), data=adj.data, col=adj.col, row=adj.row)

            # plot_superadj(adj, K=min(100, adj.shape[0]), sparse=True, labels=true_labels, vline=True, dataset="link") 

            # preds = louvain_cluster(adj, labels, random_state=seed)
            # labels = true_labels

            # from sklearn.metrics import normalized_mutual_info_score as NMI, adjusted_mutual_info_score as AMI, adjusted_rand_score as ARI, davies_bouldin_score as DBI

            # nmi = NMI(labels, preds)
            # ami = AMI(labels, preds)
            # ari = ARI(labels, preds)
            # dbi = -DBI(raw_emb, preds)

            # print(nmi, ami, ari, dbi)

            # if args.save_model:
            #     os.makedirs("saved_models", exist_ok=True)
            #     torch.save(model.state_dict(), "saved_models/Oursv-{}-{}.statedict".format(args.dataset, seed))


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







