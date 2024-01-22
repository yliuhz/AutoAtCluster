import sys
sys.path.insert(0, '.')
sys.path.append('.')

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
from utils import sparse_mx_to_torch_sparse_tensor
from node.dataset import load, load_cora_full_im, load_cora_full_diff_cls

from tqdm import tqdm

import setproctitle

import argparse
def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cora', help='type of dataset.')
    parser.add_argument('--nexp', type=int, default=10, help="Number of repeated experiments")
    return parser


# Borrowed from https://github.com/PetarV-/DGI
class GCN(nn.Module):
    def __init__(self, in_ft, out_ft, bias=True):
        super(GCN, self).__init__()
        self.fc = nn.Linear(in_ft, out_ft, bias=False)
        self.act = nn.PReLU()

        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_ft))
            self.bias.data.fill_(0.0)
        else:
            self.register_parameter('bias', None)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    # Shape of seq: (batch, nodes, features)
    def forward(self, seq, adj, sparse=False):
        seq_fts = self.fc(seq)
        if sparse:
            out = torch.unsqueeze(torch.spmm(adj, torch.squeeze(seq_fts, 0)), 0)
        else:
            out = torch.bmm(adj, seq_fts)
        if self.bias is not None:
            out += self.bias
        return self.act(out)


# Borrowed from https://github.com/PetarV-/DGI
class Readout(nn.Module):
    def __init__(self):
        super(Readout, self).__init__()

    def forward(self, seq, msk):
        if msk is None:
            return torch.mean(seq, 1)
        else:
            msk = torch.unsqueeze(msk, -1)
            return torch.mean(seq * msk, 1) / torch.sum(msk)


# Borrowed from https://github.com/PetarV-/DGI
class Discriminator(nn.Module):
    def __init__(self, n_h):
        super(Discriminator, self).__init__()
        self.f_k = nn.Bilinear(n_h, n_h, 1)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, c1, c2, h1, h2, h3, h4, s_bias1=None, s_bias2=None):
        c_x1 = torch.unsqueeze(c1, 1)
        c_x1 = c_x1.expand_as(h1).contiguous()
        c_x2 = torch.unsqueeze(c2, 1)
        c_x2 = c_x2.expand_as(h2).contiguous()

        # positive
        sc_1 = torch.squeeze(self.f_k(h2, c_x1), 2)
        sc_2 = torch.squeeze(self.f_k(h1, c_x2), 2)

        # negetive
        sc_3 = torch.squeeze(self.f_k(h4, c_x1), 2)
        sc_4 = torch.squeeze(self.f_k(h3, c_x2), 2)

        logits = torch.cat((sc_1, sc_2, sc_3, sc_4), 1)
        return logits


class Model(nn.Module):
    def __init__(self, n_in, n_h):
        super(Model, self).__init__()
        self.gcn1 = GCN(n_in, n_h)
        self.gcn2 = GCN(n_in, n_h)
        self.read = Readout()

        self.sigm = nn.Sigmoid()

        self.disc = Discriminator(n_h)

    def forward(self, seq1, seq2, adj, diff, sparse, msk, samp_bias1, samp_bias2):
        h_1 = self.gcn1(seq1, adj, sparse)
        c_1 = self.read(h_1, msk)
        c_1 = self.sigm(c_1)

        h_2 = self.gcn2(seq1, diff, sparse)
        c_2 = self.read(h_2, msk)
        c_2 = self.sigm(c_2)

        h_3 = self.gcn1(seq2, adj, sparse)
        h_4 = self.gcn2(seq2, diff, sparse)

        ret = self.disc(c_1, c_2, h_1, h_2, h_3, h_4, samp_bias1, samp_bias2)

        return ret, h_1, h_2

    def embed(self, seq, adj, diff, sparse, msk):
        h_1 = self.gcn1(seq, adj, sparse)
        c = self.read(h_1, msk)

        h_2 = self.gcn2(seq, diff, sparse)
        return (h_1 + h_2).detach(), c.detach()


class LogReg(nn.Module):
    def __init__(self, ft_in, nb_classes):
        super(LogReg, self).__init__()
        self.fc = nn.Linear(ft_in, nb_classes)
        self.sigm = nn.Sigmoid()

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq):
        ret = torch.log_softmax(self.fc(seq), dim=-1)
        return ret


def train(dataset, seed, verbose=False):

    nb_epochs = 2000
    patience = 20
    lr = 0.001
    l2_coef = 0.0
    hid_units = 512
    sparse = False

    adj, diff, features, labels, idx_train, idx_val, idx_test = load(dataset)

    ft_size = features.shape[1]
    nb_classes = np.unique(labels).shape[0]

    sample_size = 2000
    batch_size = 2

    labels = torch.LongTensor(labels)
    # idx_train = torch.LongTensor(idx_train)
    # idx_test = torch.LongTensor(idx_test)

    lbl_1 = torch.ones(batch_size, sample_size * 2)
    lbl_2 = torch.zeros(batch_size, sample_size * 2)
    lbl = torch.cat((lbl_1, lbl_2), 1)

    model = Model(ft_size, hid_units)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2_coef)

    if torch.cuda.is_available():
        model.cuda()
        labels = labels.cuda()
        lbl = lbl.cuda()
        # idx_train = idx_train.cuda()
        # idx_test = idx_test.cuda()

    b_xent = nn.BCEWithLogitsLoss()
    xent = nn.CrossEntropyLoss()
    cnt_wait = 0
    best = 1e9
    best_t = 0

    for epoch in tqdm(range(nb_epochs), total=nb_epochs):

        idx = np.random.randint(0, adj.shape[-1] - sample_size + 1, batch_size)
        ba, bd, bf = [], [], []
        for i in idx:
            ba.append(adj[i: i + sample_size, i: i + sample_size])
            bd.append(diff[i: i + sample_size, i: i + sample_size])
            bf.append(features[i: i + sample_size])

        ba = np.array(ba).reshape(batch_size, sample_size, sample_size)
        bd = np.array(bd).reshape(batch_size, sample_size, sample_size)
        bf = np.array(bf).reshape(batch_size, sample_size, ft_size)

        if sparse:
            ba = sparse_mx_to_torch_sparse_tensor(sp.coo_matrix(ba))
            bd = sparse_mx_to_torch_sparse_tensor(sp.coo_matrix(bd))
        else:
            ba = torch.FloatTensor(ba)
            bd = torch.FloatTensor(bd)

        bf = torch.FloatTensor(bf)
        idx = np.random.permutation(sample_size)
        shuf_fts = bf[:, idx, :]

        if torch.cuda.is_available():
            bf = bf.cuda()
            ba = ba.cuda()
            bd = bd.cuda()
            shuf_fts = shuf_fts.cuda()

        model.train()
        optimiser.zero_grad()

        logits, __, __ = model(bf, shuf_fts, ba, bd, sparse, None, None, None)

        loss = b_xent(logits, lbl)

        loss.backward()
        optimiser.step()

        if verbose:
            print('Epoch: {0}, Loss: {1:0.4f}'.format(epoch, loss.item()))

        if loss < best:
            best = loss
            best_t = epoch
            cnt_wait = 0
            torch.save(model.state_dict(), 'model.pkl')
        else:
            cnt_wait += 1

        if cnt_wait == patience:
            if verbose:
                print('Early stopping!')
            break

    if verbose:
        print('Loading {}th epoch'.format(best_t))
    model.load_state_dict(torch.load('model.pkl'))

    if sparse:
        adj = sparse_mx_to_torch_sparse_tensor(sp.coo_matrix(adj))
        diff = sparse_mx_to_torch_sparse_tensor(sp.coo_matrix(diff))

    features = torch.FloatTensor(features[np.newaxis])
    adj = torch.FloatTensor(adj[np.newaxis])
    diff = torch.FloatTensor(diff[np.newaxis])
    features = features.cuda()
    adj = adj.cuda()
    diff = diff.cuda()

    # embeds, _ = model.embed(features, adj, diff, sparse, None)
    with torch.no_grad():
        # embeds, _ = model.embed(features, sp_adj if sparse else adj, sparse, None)
        embeds, _ = model.embed(features, adj, diff, sparse, None)
        emb = embeds[0, :].cpu().numpy()

        import os
        os.makedirs("outputs", exist_ok=True)

        np.savez("outputs/MVGRL_{}_emb_{}.npz".format(dataset, seed), emb=emb, labels=labels.cpu().numpy())


        # from sklearn.manifold import TSNE
        # import matplotlib.pyplot as plt
        # tqdm.write("plotting")
        # tsne_z = TSNE(n_components=2, init="random").fit_transform(emb)
        # plt.figure()
        # plt.scatter(tsne_z[:, 0], tsne_z[:, 1], c=labels.cpu().numpy())
        # os.makedirs("pics", exist_ok=True)
        # plt.savefig("pics/MVGRL_{}_tsne_{}.png".format(dataset, seed))

        # from sklearn.cluster import KMeans
        # from sklearn.metrics import normalized_mutual_info_score as NMI, adjusted_mutual_info_score as AMI, adjusted_rand_score as ARI
        # tqdm.write("clustering")
        # clustering = KMeans(n_clusters=7)
        # preds = clustering.fit_predict(emb)
        # nmi = NMI(labels.cpu().numpy(), preds)
        # ami = AMI(labels.cpu().numpy(), preds)
        # ari = ARI(labels.cpu().numpy(), preds)
        # os.makedirs("results", exist_ok=True)
        # with open("results/results.txt", "a+") as f:
        #     f.write("{} {} {}\n".format(nmi, ami, ari))




    # train_embs = embeds[0, idx_train]
    # test_embs = embeds[0, idx_test]

    # train_lbls = labels[idx_train]
    # test_lbls = labels[idx_test]

    # accs = []
    # wd = 0.01 if dataset == 'citeseer' else 0.0

    # for _ in range(50):
    #     log = LogReg(hid_units, nb_classes)
    #     opt = torch.optim.Adam(log.parameters(), lr=1e-2, weight_decay=wd)
    #     log.cuda()
    #     for _ in range(300):
    #         log.train()
    #         opt.zero_grad()

    #         logits = log(train_embs)
    #         loss = xent(logits, train_lbls)

    #         loss.backward()
    #         opt.step()

    #     logits = log(test_embs)
    #     preds = torch.argmax(logits, dim=1)
    #     acc = torch.sum(preds == test_lbls).float() / test_lbls.shape[0]
    #     accs.append(acc * 100)

    # accs = torch.stack(accs)
    # print(accs.mean().item(), accs.std().item())


def train_im(rate, seed, verbose=False):

    nb_epochs = 2000
    patience = 20
    lr = 0.001
    l2_coef = 0.0
    hid_units = 512
    sparse = False

    # adj, diff, features, labels, idx_train, idx_val, idx_test = load(dataset)
    
    dataset = "cora-full"
    adj, diff, features, labels, mask = load_cora_full_im(rate, seed)

    # dataset = "ogbn-arxiv"
    # adj, diff, features, labels, mask = load_ogbn_arxiv_im(rate, seed)

    ft_size = features.shape[1]
    nb_classes = np.unique(labels).shape[0]

    sample_size = 2000
    batch_size = 2

    labels = torch.LongTensor(labels)
    # idx_train = torch.LongTensor(idx_train)
    # idx_test = torch.LongTensor(idx_test)

    lbl_1 = torch.ones(batch_size, sample_size * 2)
    lbl_2 = torch.zeros(batch_size, sample_size * 2)
    lbl = torch.cat((lbl_1, lbl_2), 1)

    model = Model(ft_size, hid_units)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2_coef)

    if torch.cuda.is_available():
        model.cuda()
        labels = labels.cuda()
        lbl = lbl.cuda()
        # idx_train = idx_train.cuda()
        # idx_test = idx_test.cuda()

    b_xent = nn.BCEWithLogitsLoss()
    xent = nn.CrossEntropyLoss()
    cnt_wait = 0
    best = 1e9
    best_t = 0

    for epoch in tqdm(range(nb_epochs), total=nb_epochs):

        idx = np.random.randint(0, adj.shape[-1] - sample_size + 1, batch_size)
        ba, bd, bf = [], [], []
        for i in idx:
            ba.append(adj[i: i + sample_size, i: i + sample_size])
            bd.append(diff[i: i + sample_size, i: i + sample_size])
            bf.append(features[i: i + sample_size])

        ba = np.array(ba).reshape(batch_size, sample_size, sample_size)
        bd = np.array(bd).reshape(batch_size, sample_size, sample_size)
        bf = np.array(bf).reshape(batch_size, sample_size, ft_size)

        if sparse:
            ba = sparse_mx_to_torch_sparse_tensor(sp.coo_matrix(ba))
            bd = sparse_mx_to_torch_sparse_tensor(sp.coo_matrix(bd))
        else:
            ba = torch.FloatTensor(ba)
            bd = torch.FloatTensor(bd)

        bf = torch.FloatTensor(bf)
        idx = np.random.permutation(sample_size)
        shuf_fts = bf[:, idx, :]

        if torch.cuda.is_available():
            bf = bf.cuda()
            ba = ba.cuda()
            bd = bd.cuda()
            shuf_fts = shuf_fts.cuda()

        model.train()
        optimiser.zero_grad()

        logits, __, __ = model(bf, shuf_fts, ba, bd, sparse, None, None, None)

        loss = b_xent(logits, lbl)

        loss.backward()
        optimiser.step()

        if verbose:
            print('Epoch: {0}, Loss: {1:0.4f}'.format(epoch, loss.item()))

        if loss < best:
            best = loss
            best_t = epoch
            cnt_wait = 0
            torch.save(model.state_dict(), 'model.pkl')
        else:
            cnt_wait += 1

        if cnt_wait == patience:
            if verbose:
                print('Early stopping!')
            break

    if verbose:
        print('Loading {}th epoch'.format(best_t))
    model.load_state_dict(torch.load('model.pkl'))

    if sparse:
        adj = sparse_mx_to_torch_sparse_tensor(sp.coo_matrix(adj))
        diff = sparse_mx_to_torch_sparse_tensor(sp.coo_matrix(diff))

    features = torch.FloatTensor(features[np.newaxis])
    adj = torch.FloatTensor(adj[np.newaxis])
    diff = torch.FloatTensor(diff[np.newaxis])
    features = features.cuda()
    adj = adj.cuda()
    diff = diff.cuda()

    # embeds, _ = model.embed(features, adj, diff, sparse, None)
    with torch.no_grad():
        # embeds, _ = model.embed(features, sp_adj if sparse else adj, sparse, None)
        embeds, _ = model.embed(features, adj, diff, sparse, None)
        emb = embeds[0, :].cpu().numpy()

        import os
        os.makedirs("outputs", exist_ok=True)

        np.savez("outputs/MVGRL_{}_emb_{:.1f}_{}_l.npz".format(dataset, rate, seed), emb=emb, labels=labels.cpu().numpy())


        # from sklearn.manifold import TSNE
        # import matplotlib.pyplot as plt
        # tqdm.write("plotting")
        # tsne_z = TSNE(n_components=2, init="random").fit_transform(emb)
        # plt.figure()
        # plt.scatter(tsne_z[:, 0], tsne_z[:, 1], c=labels.cpu().numpy())
        # os.makedirs("pics", exist_ok=True)
        # plt.savefig("pics/MVGRL_{}_tsne_{}.png".format(dataset, seed))

        # from sklearn.cluster import KMeans
        # from sklearn.metrics import normalized_mutual_info_score as NMI, adjusted_mutual_info_score as AMI, adjusted_rand_score as ARI
        # tqdm.write("clustering")
        # clustering = KMeans(n_clusters=7)
        # preds = clustering.fit_predict(emb)
        # nmi = NMI(labels.cpu().numpy(), preds)
        # ami = AMI(labels.cpu().numpy(), preds)
        # ari = ARI(labels.cpu().numpy(), preds)
        # os.makedirs("results", exist_ok=True)
        # with open("results/results.txt", "a+") as f:
        #     f.write("{} {} {}\n".format(nmi, ami, ari))




    # train_embs = embeds[0, idx_train]
    # test_embs = embeds[0, idx_test]

    # train_lbls = labels[idx_train]
    # test_lbls = labels[idx_test]

    # accs = []
    # wd = 0.01 if dataset == 'citeseer' else 0.0

    # for _ in range(50):
    #     log = LogReg(hid_units, nb_classes)
    #     opt = torch.optim.Adam(log.parameters(), lr=1e-2, weight_decay=wd)
    #     log.cuda()
    #     for _ in range(300):
    #         log.train()
    #         opt.zero_grad()

    #         logits = log(train_embs)
    #         loss = xent(logits, train_lbls)

    #         loss.backward()
    #         opt.step()

    #     logits = log(test_embs)
    #     preds = torch.argmax(logits, dim=1)
    #     acc = torch.sum(preds == test_lbls).float() / test_lbls.shape[0]
    #     accs.append(acc * 100)

    # accs = torch.stack(accs)
    # print(accs.mean().item(), accs.std().item())


def train_diff_cls(nclass, seed, verbose=False):

    nb_epochs = 2000
    patience = 20
    lr = 0.001
    l2_coef = 0.0
    hid_units = 512
    sparse = False

    # adj, diff, features, labels, idx_train, idx_val, idx_test = load(dataset)
    
    dataset = "cora-full"
    adj, diff, features, labels, mask = load_cora_full_diff_cls(nclass, seed)

    num_nodes, ft_size = features.shape
    nb_classes = np.unique(labels).shape[0]

    sample_size = min(num_nodes, 2000)
    batch_size = 2

    labels = torch.LongTensor(labels)
    # idx_train = torch.LongTensor(idx_train)
    # idx_test = torch.LongTensor(idx_test)

    lbl_1 = torch.ones(batch_size, sample_size * 2)
    lbl_2 = torch.zeros(batch_size, sample_size * 2)
    lbl = torch.cat((lbl_1, lbl_2), 1)

    model = Model(ft_size, hid_units)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2_coef)

    if torch.cuda.is_available():
        model.cuda()
        labels = labels.cuda()
        lbl = lbl.cuda()
        # idx_train = idx_train.cuda()
        # idx_test = idx_test.cuda()

    b_xent = nn.BCEWithLogitsLoss()
    xent = nn.CrossEntropyLoss()
    cnt_wait = 0
    best = 1e9
    best_t = 0

    for epoch in tqdm(range(nb_epochs), total=nb_epochs):

        idx = np.random.randint(0, adj.shape[-1] - sample_size + 1, batch_size)
        ba, bd, bf = [], [], []
        for i in idx:
            ba.append(adj[i: i + sample_size, i: i + sample_size])
            bd.append(diff[i: i + sample_size, i: i + sample_size])
            bf.append(features[i: i + sample_size])

        ba = np.array(ba).reshape(batch_size, sample_size, sample_size)
        bd = np.array(bd).reshape(batch_size, sample_size, sample_size)
        bf = np.array(bf).reshape(batch_size, sample_size, ft_size)

        if sparse:
            ba = sparse_mx_to_torch_sparse_tensor(sp.coo_matrix(ba))
            bd = sparse_mx_to_torch_sparse_tensor(sp.coo_matrix(bd))
        else:
            ba = torch.FloatTensor(ba)
            bd = torch.FloatTensor(bd)

        bf = torch.FloatTensor(bf)
        idx = np.random.permutation(sample_size)
        shuf_fts = bf[:, idx, :]

        if torch.cuda.is_available():
            bf = bf.cuda()
            ba = ba.cuda()
            bd = bd.cuda()
            shuf_fts = shuf_fts.cuda()

        model.train()
        optimiser.zero_grad()

        logits, __, __ = model(bf, shuf_fts, ba, bd, sparse, None, None, None)

        loss = b_xent(logits, lbl)

        loss.backward()
        optimiser.step()

        if verbose:
            print('Epoch: {0}, Loss: {1:0.4f}'.format(epoch, loss.item()))

        if loss < best:
            best = loss
            best_t = epoch
            cnt_wait = 0
            torch.save(model.state_dict(), 'model.pkl')
        else:
            cnt_wait += 1

        if cnt_wait == patience:
            if verbose:
                print('Early stopping!')
            break

    if verbose:
        print('Loading {}th epoch'.format(best_t))
    model.load_state_dict(torch.load('model.pkl'))

    if sparse:
        adj = sparse_mx_to_torch_sparse_tensor(sp.coo_matrix(adj))
        diff = sparse_mx_to_torch_sparse_tensor(sp.coo_matrix(diff))

    features = torch.FloatTensor(features[np.newaxis])
    adj = torch.FloatTensor(adj[np.newaxis])
    diff = torch.FloatTensor(diff[np.newaxis])
    features = features.cuda()
    adj = adj.cuda()
    diff = diff.cuda()

    # embeds, _ = model.embed(features, adj, diff, sparse, None)
    with torch.no_grad():
        # embeds, _ = model.embed(features, sp_adj if sparse else adj, sparse, None)
        embeds, _ = model.embed(features, adj, diff, sparse, None)
        emb = embeds[0, :].cpu().numpy()

        import os
        os.makedirs("outputs", exist_ok=True)

        np.savez("outputs/MVGRL_{}_emb_{:d}_{}.npz".format(dataset, nclass, seed), emb=emb, labels=labels.cpu().numpy())


        # from sklearn.manifold import TSNE
        # import matplotlib.pyplot as plt
        # tqdm.write("plotting")
        # tsne_z = TSNE(n_components=2, init="random").fit_transform(emb)
        # plt.figure()
        # plt.scatter(tsne_z[:, 0], tsne_z[:, 1], c=labels.cpu().numpy())
        # os.makedirs("pics", exist_ok=True)
        # plt.savefig("pics/MVGRL_{}_tsne_{}.png".format(dataset, seed))

        # from sklearn.cluster import KMeans
        # from sklearn.metrics import normalized_mutual_info_score as NMI, adjusted_mutual_info_score as AMI, adjusted_rand_score as ARI
        # tqdm.write("clustering")
        # clustering = KMeans(n_clusters=7)
        # preds = clustering.fit_predict(emb)
        # nmi = NMI(labels.cpu().numpy(), preds)
        # ami = AMI(labels.cpu().numpy(), preds)
        # ari = ARI(labels.cpu().numpy(), preds)
        # os.makedirs("results", exist_ok=True)
        # with open("results/results.txt", "a+") as f:
        #     f.write("{} {} {}\n".format(nmi, ami, ari))




    # train_embs = embeds[0, idx_train]
    # test_embs = embeds[0, idx_test]

    # train_lbls = labels[idx_train]
    # test_lbls = labels[idx_test]

    # accs = []
    # wd = 0.01 if dataset == 'citeseer' else 0.0

    # for _ in range(50):
    #     log = LogReg(hid_units, nb_classes)
    #     opt = torch.optim.Adam(log.parameters(), lr=1e-2, weight_decay=wd)
    #     log.cuda()
    #     for _ in range(300):
    #         log.train()
    #         opt.zero_grad()

    #         logits = log(train_embs)
    #         loss = xent(logits, train_lbls)

    #         loss.backward()
    #         opt.step()

    #     logits = log(test_embs)
    #     preds = torch.argmax(logits, dim=1)
    #     acc = torch.sum(preds == test_lbls).float() / test_lbls.shape[0]
    #     accs.append(acc * 100)

    # accs = torch.stack(accs)
    # print(accs.mean().item(), accs.std().item())


if __name__ == '__main__':
    import warnings
    warnings.filterwarnings("ignore")
    torch.cuda.set_device(0)


    parser = make_parser()
    args = parser.parse_args()

    # 'cora', 'citeseer', 'pubmed'
    dataset = args.dataset

    seeds = np.arange(0, args.nexp, dtype=int)
    for seed in tqdm(seeds, total=args.nexp):
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        setproctitle.setproctitle("MVGRL-{}-{}".format(args.dataset, seed))

        train(dataset, seed)

        # rates = np.arange(0.1, 1.0, 0.2) # minimum rention rates
        # for rate in rates:
        #     setproctitle.setproctitle("MVGRLim-{:.1f}-{:d}".format(rate, seed))

        #     train_im(rate, seed, verbose=True)


        # nclasses = np.arange(2, 12, 2, dtype=int)
        # for nclass in nclasses:
        #     setproctitle.setproctitle("MVGRLdc-{:d}-{:d}".format(nclass, seed))

        #     train_diff_cls(nclass, seed, verbose=True)


