import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

import dgl
import dgl.nn as dnn

from utils import idx_to_coord

class GCN_(nn.Module):
    def __init__(self, in_ft, out_ft, act=F.relu):
        super(GCN, self).__init__()
        self.act = act
        self.weight = Parameter(torch.FloatTensor(in_ft, out_ft))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight, gain=1.414)

    # Shape of seq: (nodes, features)
    def forward(self, seq, adj):
        support = torch.mm(seq, self.weight)
        out = torch.spmm(adj, support)
        out = self.act(out)
        
        return out

class GCN(nn.Module):
    def __init__(self, in_ft, out_ft, act=F.relu, bias=True, bn=False):
        super(GCN, self).__init__()
        self.fc = nn.Linear(in_ft, out_ft, bias=False)
        self.act = nn.PReLU() if act == 'prelu' else act
        
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_ft))
            self.bias.data.fill_(0.0)
        else:
            self.register_parameter('bias', None)

        self.bn_s = bn
        self.bn = nn.BatchNorm1d(out_ft)

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

        if self.bn_s:
            out = self.bn(out.squeeze(0)).unsqueeze(0)
        
        return self.act(out)

class GAE(nn.Module):
    def __init__(self, gnnlayers, dims, dropout=0.):
        super(GAE, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(gnnlayers-1):
            # self.layers.append(GCN(dims[i], dims[i+1]))
            self.layers.append(dnn.GraphConv(dims[i], dims[i+1], norm="both", activation=F.relu))
        # self.layers.append(GCN(dims[-2], dims[-1], act=lambda x: x))
        self.layers.append(dnn.GraphConv(dims[-2], dims[-1], activation=lambda x: x))

        # self.fc = nn.Linear(dims[-1], dims[0])

        self.dropout = dropout

    def encoder(self, g, x):
        z = x
        for layer in self.layers:
            z = layer(g, z)
        return z

    def decoder(self, z, act=lambda x: x):
        z = F.dropout(z, self.dropout, training=self.training)
        adj_rec = act(torch.spmm(z, z.t()))
        # adj_rec = act(torch.sparse.mm(z, z.t()))
        return adj_rec

    def forward(self, g, x):
        z = self.encoder(g, x)
        adj_rec = self.decoder(z)
        return adj_rec

    def loss(self, adj, adj_rec, pos_weight=None):
        device="cpu"

        if pos_weight is not None:
            self.loss_function = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
        else:
            self.loss_function = nn.BCEWithLogitsLoss()
            
        return self.loss_function(adj_rec.to(device).view(-1), adj.to(device).view(-1))

class DGI(nn.Module):
    def __init__(self, n_in, n_h, activation, bn=False):
        super(DGI, self).__init__()
        self.gcn = GCN(n_in, n_h, activation, bn=bn)
        self.read = AvgReadout()

        self.sigm = nn.Sigmoid()

        self.disc = Discriminator(n_h)

    def forward(self, seq1, seq2, adj, sparse, msk, samp_bias1, samp_bias2):
        h_1 = self.gcn(seq1, adj, sparse)

        c = self.read(h_1, msk)
        c = self.sigm(c)

        h_2 = self.gcn(seq2, adj, sparse)

        ret = self.disc(c, h_1, h_2, samp_bias1, samp_bias2)

        return ret

    # Detach the return variables
    def encoder(self, seq, adj, sparse, msk):
        h_1 = self.gcn(seq, adj, sparse)
        c = self.read(h_1, msk)

        return h_1.detach(), c.detach()

    def loss(self, logits, labels):
        loss_function = nn.BCEWithLogitsLoss()
        return loss_function(logits, labels)

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

    def forward(self, c, h_pl, h_mi, s_bias1=None, s_bias2=None):
        c_x = torch.unsqueeze(c, 1)
        c_x = c_x.expand_as(h_pl)

        sc_1 = torch.squeeze(self.f_k(h_pl, c_x), 2)
        sc_2 = torch.squeeze(self.f_k(h_mi, c_x), 2)

        if s_bias1 is not None:
            sc_1 += s_bias1
        if s_bias2 is not None:
            sc_2 += s_bias2

        logits = torch.cat((sc_1, sc_2), 1)

        return logits

    def forward_MVGRL(self, c1, c2, h1, h2, h3, h4, s_bias1=None, s_bias2=None):
        c_x1 = torch.unsqueeze(c1, 1) # 图的表征
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

class AvgReadout(nn.Module):
    def __init__(self):
        super(AvgReadout, self).__init__()

    def forward(self, seq, msk):
        if msk is None:
            return torch.mean(seq, 1)
        else:
            msk = torch.unsqueeze(msk, -1)
            return torch.sum(seq * msk, 1) / torch.sum(msk)   


class MVGRL(nn.Module):
    def __init__(self, n_in, n_h):
        super(MVGRL, self).__init__()
        self.gcn1 = GCN(n_in, n_h)
        self.gcn2 = GCN(n_in, n_h)
        self.read = AvgReadout()

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

        ret = self.disc.forward_MVGRL(c_1, c_2, h_1, h_2, h_3, h_4, samp_bias1, samp_bias2)

        return ret, h_1, h_2

    def encoder(self, seq, adj, diff, sparse, msk):
        h_1 = self.gcn1(seq, adj, sparse)
        c = self.read(h_1, msk)

        h_2 = self.gcn2(seq, diff, sparse)
        return (h_1 + h_2).detach(), c.detach()


if __name__ == "__main__":

    model = GAE(2, [512,512])
    