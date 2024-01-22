
from turtle import update
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import scipy.sparse as sp
from vis import plot_superadj
import matplotlib.pyplot as plt
from utils import louvain_cluster, make_parser
from load import load_assortative, load_syn_subgraph

from sklearn.preprocessing import normalize
from sklearn.manifold import TSNE
from tqdm import tqdm
from sklearn.metrics import adjusted_rand_score as ARI
import os

from tqdm import tqdm

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)

import setproctitle

class BinaryClassification(nn.Module):
    def __init__(self, in_dim, hidden_dim, dropout=0.1):
        super(BinaryClassification, self).__init__()
        # Number of input features is 12.
        self.layer_1 = nn.Linear(in_dim, hidden_dim) 
        self.layer_2 = nn.Linear(hidden_dim, hidden_dim)
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
        x = self.relu(self.layer_1(inputs))
        x = self.batchnorm1(x)
        x = self.relu(self.layer_2(x))
        x = self.batchnorm2(x)
        x = self.dropout(x)
        x = self.layer_out(x)
        
        return x.reshape(-1)

def update_similarity(emb, pos_num, neg_num, test=False):
    # sim = normalize(emb)
    sim = emb
    sim = np.matmul(sim, np.transpose(sim))
    # sim = sim - np.diag(np.diag(sim))
    # print("plotting ...")
    # sim = sim.reshape(-1)
    # sim_ids = np.argsort(-sim)
    # sim = sim[sim_ids]
    # plt.figure()
    # plt.scatter([x for x in range(sim.shape[0])], sim)
    # plt.savefig("pics/sim.png")
    # exit(0)

    # pos_inds = np.argpartition(-(sim - np.diag(np.diag(sim))).reshape(-1), pos_num)[:pos_num]
    pos_inds = np.argpartition(-sim.reshape(-1), pos_num)[:pos_num]
    neg_inds = np.argpartition(sim.reshape(-1), neg_num)[:neg_num]

    train_inds = np.hstack([pos_inds, neg_inds])
    test_inds = np.setdiff1d(np.arange(emb.shape[0]*emb.shape[0], dtype=np.int64), train_inds)
    
    if test:
        return torch.LongTensor(pos_inds), torch.LongTensor(neg_inds), torch.LongTensor(test_inds)
    return torch.LongTensor(pos_inds), torch.LongTensor(neg_inds)


def update(sim, pos_t, neg_t):
    # sim = sim.reshape(-1)

    arr = np.arange(sim.shape[0], dtype=int)
    pos_inds = arr[sim >= pos_t]
    neg_inds = arr[sim < neg_t]

    return torch.LongTensor(pos_inds), torch.LongTensor(neg_inds)


def select_valid(b_sim, b_adj, rate=0.1):
    b_adj = b_adj.toarray()

    for i, j in b_adj.nonzeros():
        pass




if __name__ == "__main__":
    parser = make_parser()
    parser.add_argument("--model", type=str, default="GAE")
    # parser.add_argument("--pos", type=float, default=0.01)
    # parser.add_argument("--neg", type=float, default=0.9)
    args = parser.parse_args()
    print(args)

    dataset = args.dataset
    print("dataset={}".format(dataset))
    results_lo, results_km = [], []

    emb_paths = {
        "GAE": "emb_models/GAE/outputs", 
        "VGAE": "emb_models/GAE/outputs",
        "ARGA": "emb_models/ARGA/ARGA/arga/outputs",
        "ARVGA": "emb_models/ARGA/ARGA/arga/outputs",
        "AGE": "emb_models/AGE/outputs",
        "DGI": "emb_models/DGI/outputs",
        "MVGRL": "emb_models/MVGRL/outputs",
        "GRACE": "emb_models/GRACE/outputs",
        "GGD": "emb_models/GGD/manual_version/outputs"
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

        setproctitle.setproctitle("lkO3-{}-{}-{}".format(args.model, dataset[:2], seed))

        if args.model == "GRACE":
            dataset_ = GRACE_datasets[dataset]
        else:
            dataset_ = dataset

        emb_path = os.path.join(emb_paths[args.model], "{}_{}_emb_{}.npz".format(args.model, dataset_, seed))
        data = np.load(emb_path)
        adj, _, true_labels = load_assortative(dataset=dataset)
        # adj, _, _ = load_syn_subgraph(dataset=dataset)

        # emb, true_labels = data["emb"], data["labels"]
        emb = data["emb"]
        # zmax = emb.max(axis=1, keepdims=True)
        # zmin = emb.min(axis=1, keepdims=True)
        # emb = (emb - zmin) / (zmax - zmin)

        print('Emb.shape={}'.format(emb.shape))

        model_config = {
            "in_dim": emb.shape[-1],
            "hidden_dim": 256,
            "epochs": 200, # 800 
            "lr": 1e-5,
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

        model = BinaryClassification(model_config["in_dim"], model_config["hidden_dim"]).to(model_config["device"])
        optimizer = optim.Adam(model.parameters(), lr=model_config["lr"], eps=1e-3)
        criterion = nn.BCEWithLogitsLoss()
        print(model)

        best_loss = 100
        early = 0
        # sim = normalize(emb)
        # sim = np.matmul(sim, np.transpose(sim))
        # sim_argsort = np.argsort(sim)

        # pos_num = model_config["sample_size"]
        data = np.load("params_O3/{}-{}.npz".format(args.model, dataset))
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

                # plt.figure()
                # plot_tsne = True
                # if not plot_tsne:
                #     plt.scatter([x for x in range(sampled_inds.shape[0])], b_sim[sampled_inds.cpu()])
                #     plt.axvline(x = pos_num, color="red")
                # else:
                #     tsne_z = TSNE(n_components=2).fit_transform(b_emb)
                #     plt.scatter(tsne_z[:, 0], tsne_z[:, 1], c=b_sim.reshape(model_config["batch_size"], -1)[0])
                #     plt.colorbar()
                #     plt.annotate("0", (tsne_z[0, 0], tsne_z[0, 1]))
                # plt.savefig("pics/sims.png")

                # exit(0)

                xinds = sampled_inds // emb.shape[0]
                yinds = sampled_inds % emb.shape[0]

                # x = b_emb[pos_inds]
                # y = b_emb[neg_inds]

                x = torch.index_select(inx, 0, xinds)
                y = torch.index_select(inx, 0, yinds)

                # x = torch.FloatTensor(x).to(model_config["device"])
                # y = torch.FloatTensor(y).to(model_config["device"])

                labels = torch.cat([torch.ones(ed - st), torch.zeros(ed - st)]).to(model_config["device"])
                # inputs = (x * y).to(model_config["device"])
                inputs = (x - y).abs().to(model_config["device"])
                # zmax = inputs.max(dim=1, keepdim=True)[0]
                # zmin = inputs.min(dim=1, keepdim=True)[0]
                # inputs = (inputs - zmin) / (zmax - zmin)
                inputs = F.normalize(inputs)

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

            row, col, data = [], [], []
            arr = np.arange(emb.shape[0], dtype=int)
            M_edges = int(np.log(emb.shape[0]))
            M_edges = emb.shape[0]
            print("M_edges={}".format(M_edges))

            # NOTE: N*LOGN edges
            for i in tqdm(range(emb.shape[0]), total=emb.shape[0]):
                # inputs = (inx[i] * inx).to(model_config["device"])
                # inputs = ((inx[i] + inx) / 2).to(model_config["device"])
                inputs = (inx[i] - inx).abs().to(model_config["device"])

                # zmax = inputs.max(dim=1, keepdim=True)[0]
                # zmin = inputs.min(dim=1, keepdim=True)[0]
                # inputs = (inputs - zmin) / (zmax - zmin)
                inputs = F.normalize(inputs)

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

            tqdm.write('n={}, m={}'.format(adj.shape[0], adj.sum()))

            # np.savez("data/{}_subgraph/link.npz".format(dataset), adj=adj.toarray())

            # plot_superadj(adj, K=min(100, adj.shape[0]), sparse=True, labels=true_labels, vline=True, dataset="link")


            # tqdm.write("KMeans")
            # from sklearn.cluster import KMeans
            # preds = KMeans(n_clusters=7).fit_predict(emb)
            # ari_km = ARI(true_labels, preds)
            # results_km.append(ari_km)
            # tqdm.write("Louvain clustering")
            # preds = louvain_cluster(adj, true_labels, random_state=seed)
            # ari_lo = ARI(true_labels, preds)
            # results_lo.append(ari_lo)
            

            os.makedirs("link_adj_O3/{}".format(args.model), exist_ok=True)
            np.savez("link_adj_O3/{}/{}_{}.npz".format(args.model, dataset, seed), data=adj.data, col=adj.col, row=adj.row)


            if args.save_model:
                os.makedirs("saved_models_O3", exist_ok=True)
                torch.save(model.state_dict(), "saved_models_O3/{}-{}-{}.statedict".format(args.model, args.dataset, seed))


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







