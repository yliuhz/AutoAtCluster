
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
from sklearn.cluster import KMeans
from tqdm import tqdm
from sklearn.metrics import adjusted_rand_score as ARI
import os

from tqdm import tqdm
import setproctitle

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)

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
    
    if test:
        train_inds = np.hstack([pos_inds, neg_inds])
        test_inds = np.setdiff1d(np.arange(emb.shape[0]*emb.shape[0], dtype=np.int64), train_inds)
        return torch.LongTensor(pos_inds), torch.LongTensor(neg_inds), torch.LongTensor(test_inds)
    return torch.LongTensor(pos_inds), torch.LongTensor(neg_inds)


def update(sim, pos_t, neg_t):
    # sim = sim.reshape(-1)

    nn = np.arange(sim.shape[0], dtype=int)
    pos_inds = nn[sim >= pos_t]
    neg_inds = nn[sim < neg_t]

    return torch.LongTensor(pos_inds), torch.LongTensor(neg_inds)


def select_valid(b_sim, b_adj, rate=0.1):
    # b_adj = sp.coo_matrix(b_adj)
    # n = b_adj.shape[0]
    # m = b_adj.sum()
    # ret_m = round(m * rate)
    # ids = b_adj.nonzero()
    # ids = ids[0] * n + ids[1]
    # ids = np.random.permutation(ids)
    # pos_inds = ids[:ret_m]

    # b_sim = b_sim.reshape(-1)
    # neg_inds = np.argpartition(b_sim, m)[:ret_m]
    # pos_inds = np.argpartition(-b_sim, ret_m)[:ret_m]

    n = b_adj.shape[0]
    m = b_adj.sum()
    ret_m = round(m * rate)
    # arr = np.random.permutation(n*n)[:2*ret_m]
    arr = np.random.choice(n*n, 2*ret_m)
    
    pos_inds, neg_inds = arr[:ret_m], arr[ret_m:]

    return torch.LongTensor(pos_inds), torch.LongTensor(neg_inds)

def acc(preds, labels):
    preds = torch.sigmoid(preds) > 0.5
    return (preds == labels).sum() / labels.shape[0]


def average_distance(preds):
    preds = torch.sigmoid(preds)
    dist =  (preds - 0.5).abs().mean()
    ret = torch.exp(-dist)
    return ret

def average_distance_2(preds):
    preds = torch.sigmoid(preds)
    dist = -(preds - 0.5).abs() + 0.5
    return dist.mean()



if __name__ == "__main__":
    parser = make_parser()
    parser.add_argument("--model", type=str, default="GAE")
    args = parser.parse_args()
    print(args)

    dataset = args.dataset
    print("dataset={}".format(dataset))
    device = "cuda:0"

    setproctitle.setproctitle("gs-{}-{}".format(args.model, dataset[:2]))

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
        "wiki": 19,
        "amazon-photo": 8,
        "amazon-computers": 10,
    }

    seeds = np.arange(0, args.nexp, dtype=int) # seed = 0
    # seeds = [11]
    # seeds = [7]
    for seed in tqdm(seeds, total=len(seeds)):
        np.random.seed(seed)
        torch.manual_seed(seed)

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
        inx = torch.FloatTensor(emb).to(device)

        km_preds = KMeans(n_clusters=nclasses[args.dataset]).fit_predict(emb)
        ari_km = ARI(true_labels, km_preds)

        sim = normalize(emb)
        sim = np.matmul(sim, np.transpose(sim))

        print('Emb.shape={}'.format(emb.shape))

        valid_pos_inds, valid_neg_inds = select_valid(sim, adj, rate=0.5)
        valid_inds = torch.cat((valid_pos_inds, valid_neg_inds), 0).to(device)
        valid_x = valid_inds // emb.shape[0]
        valid_y = valid_inds % emb.shape[0]
        x = torch.index_select(inx, 0, valid_x)
        y = torch.index_select(inx, 0, valid_y)
        # valid_inputs = (x * y).to(device)
        valid_inputs = ((x + y) / 2).to(device)
        valid_inputs = F.normalize(valid_inputs)
        valid_labels = torch.cat([torch.ones(valid_pos_inds.shape[0]), torch.zeros(valid_neg_inds.shape[0])]).to(device)

        pos_rs = [1e-3, 5e-3, 1e-2]
        # pos_rs = [0.011]
        neg_rs = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
        # neg_rs = [0.9]
        reports = []
        reports_ari = []
        for pos_r in pos_rs:
            for neg_r in neg_rs:
                print("pos_r:{}, neg_r:{}".format(pos_r, neg_r))

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

                # inx = torch.FloatTensor(emb).to(model_config["device"])
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
                # pos_num = model_config["m"] * pos_r
                pos_num = round(emb.shape[0] * emb.shape[0] * pos_r)
                # neg_num = pos_num * 5 # pos_num * 5
                neg_num = round(emb.shape[0] * emb.shape[0] * neg_r)
                pos_inds, neg_inds = update_similarity(normalize(emb), pos_num, neg_num)
                # pos_inds = np.setdiff1d(pos_inds.numpy(), valid_pos_inds.numpy())
                # neg_inds = np.setdiff1d(neg_inds.numpy(), valid_neg_inds.numpy())
                # pos_inds = torch.LongTensor(pos_inds)
                # neg_inds = torch.LongTensor(neg_inds)

                model_config["sample_size"] = min(model_config["sample_size"], pos_inds.shape[0])


                for epoch in tqdm(range(model_config["epochs"]), total=model_config["epochs"]):
                    model.train()

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

                        # x = torch.FloatTensor(x).to(model_config["device"])
                        # y = torch.FloatTensor(y).to(model_config["device"])

                        labels = torch.cat([torch.ones(ed - st), torch.zeros(ed - st)]).to(model_config["device"])
                        # inputs = (x * y).to(model_config["device"])
                        inputs = ((x + y) / 2).to(model_config["device"])
                        # zmax = inputs.max(dim=1, keepdim=True)[0]
                        # zmin = inputs.min(dim=1, keepdim=True)[0]
                        # diff = zmax - zmin
                        # diff[diff == 0] = 1.
                        # inputs = (inputs - zmin) / diff
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

                    if loss > best_loss:
                        early += 1
                    else:
                        best_loss = loss
                        early = 0
                    
                    if early > model_config["patience"]:
                        tqdm.write("Early stop at {} epoch".format(epoch))
                        break


                    with torch.no_grad():
                        model.eval()

                        valid_preds = model(valid_inputs)
                        valid_loss = criterion(valid_preds, valid_labels)
                        valid_acc = acc(valid_preds, valid_labels)
                        valid_dist = average_distance(valid_preds)
                        # print(torch.sigmoid(valid_preds))

                        if epoch % model_config["print_epoch"] == 0:
                            tqdm.write("EPOCH {:d}/{:d} Loss:{:.4f} Valid Loss:{:.4f} Valid Acc:{:.4f} Valid Dist:{:.4f}".format(epoch+1, model_config["epochs"], loss.item(), valid_loss.item(), valid_acc.item(), valid_dist.item()))
                with torch.no_grad():
                    model.eval()

                    valid_preds = model(valid_inputs)
                    valid_loss = criterion(valid_preds, valid_labels)
                    # valid_dist = average_distance(valid_preds)
                    valid_dist = average_distance_2(valid_preds)
                    # reports.append(valid_loss.item())
                    reports.append(valid_dist.item())

                    # TIPS: old
                    row, col, data = [], [], []
                    nnr = np.arange(emb.shape[0], dtype=int)
                    M_edges = int(np.log(emb.shape[0]))
                    M_edges = emb.shape[0]
                    # print("M_edges={}".format(M_edges))

                    # NOTE: N*LOGN edges
                    for i in tqdm(range(emb.shape[0]), total=emb.shape[0]):
                        inputs = (inx[i] * inx).to(model_config["device"])

                        # zmax = inputs.max(dim=1, keepdim=True)[0]
                        # zmin = inputs.min(dim=1, keepdim=True)[0]
                        # inputs = (inputs - zmin) / (zmax - zmin)
                        inputs = F.normalize(inputs)

                        preds = torch.sigmoid(model(inputs))

                        # preds[preds > 0.5] = 1.
                        # preds[preds <= 0.5] = 0.
                        neib_ids = nnr[preds.cpu() > 0.5]
                        if neib_ids.shape[0] > M_edges:
                            neib_ids = np.random.permutation(neib_ids)
                            neib_ids = neib_ids[:M_edges]

                        # data += (preds.cpu().tolist())
                        # row += ([i] * emb.shape[0])
                        # col += ([x for x in range(emb.shape[0])])

                        data += [1] * neib_ids.shape[0]
                        row += [i] * neib_ids.shape[0]
                        col += neib_ids.tolist()

                    adj_ = sp.coo_matrix((data, (row, col)), shape=(emb.shape[0], emb.shape[0]))
                    adj_.eliminate_zeros()

                    tqdm.write('n={}, m={}'.format(adj_.shape[0], adj_.sum()))

                    # np.savez("data/{}_subgraph/link.npz".format(dataset), adj=adj.toarray())

                    # plot_superadj(adj, K=min(100, adj.shape[0]), sparse=True, labels=true_labels, vline=True, dataset="link")

                    # tqdm.write("Louvain clustering")
                    # preds = louvain_cluster(adj, true_labels, random_state=seed)
                    # ari_lo = ARI(true_labels, preds)
                    # reports_ari.append(ari_lo)
                    reports_ari.append(0)
                    

                    # os.makedirs("link_adj/{}".format(args.model), exist_ok=True)
                    # np.savez("link_adj/{}/{}_{}.npz".format(args.model, dataset, seed), data=adj.data, col=adj.col, row=adj.row)

        idx = 0
        for pos_r in pos_rs:
            for neg_r in neg_rs:
                print("pos_r:{}, neg_r:{}, v_loss:{:.4f}, ari_lo:{:.4f}, ari_km:{:.4f}".format(pos_r, neg_r, reports[idx], reports_ari[idx], ari_km))
                idx += 1
        

        print("="*50)
        idx = np.argmin(reports)
        x = idx // len(neg_rs)
        y = idx % len(neg_rs)
        print("Best params: pos_r:{}, neg_r:{}, v_loss:{:.4f}, ari_lo:{:.4f}, ari_km:{:.4f}".format(pos_rs[x], neg_rs[y], reports[idx], reports_ari[idx], ari_km))

        os.makedirs("params_O1", exist_ok=True)
        np.savez("params_O1/{}-{}.npz".format(args.model, args.dataset), pos=pos_rs[x], neg=neg_rs[y])


                





