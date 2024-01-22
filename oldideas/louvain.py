
from community import community_louvain
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_networkx
import numpy as np
from sklearn.preprocessing import normalize
import torch
import argparse

parser = argparse.ArgumentParser(description="louvain algorithm")
parser.add_argument('--model', '-m', default='', type=str, required=True)
args = parser.parse_args()

dataset_ = 'cora'
exp = 0

dataset = Planetoid(root='./data/cora_planetoid', name=dataset_)
data = dataset[0]

N = 10

for exp in range(N):
    G = to_networkx(data, to_undirected=True)
    print('Number of nodes={}'.format(G.number_of_nodes()))
    print('Number of edges={}'.format(G.number_of_edges()))

    if args.model == 'GDCL':
        print('using GDCL')
        data_ = np.load('GDCL_cora.npz')
        emb = data_['z']
    else:
        data_ = np.load('data/qual_{}_age_{}.npz'.format(dataset_, exp))
        emb = data_['z']

    emb0 = emb

    emb = normalize(emb)

    emb1 = np.matmul(emb, np.transpose(emb)) + 1. # X * X^T + 1.
    emb2 = 0. 

    emb = torch.sigmoid(torch.FloatTensor(emb1)).numpy()


    # G2 = nx.Graph()
    # G2.add_edge(0, 1, weight=0.5)
    # G2.add_edge(0, 1, weight=0.7)
    # print(G2.edges.data())

    # for idx, e in enumerate(G.edges()):
    #     G.add_edge(e[0], e[1], weight=emb[e])

    for i in range(emb.shape[0]):
        for j in range(emb.shape[1]):
            G.add_edge(i, j, weight=emb[i,j])

    partition = community_louvain.best_partition(G)

        
    print(len(partition))
    # print(set(partition.values()))
    print('Number of clusters={}'.format(len(set(partition.values()))))
    print(list(partition.items())[0])

    assert list(partition.keys()) == sorted(list(partition.keys()))
    with open('{}_cluster_{}.txt'.format(dataset_, exp), 'w') as f:
        for key, value in partition.items():
            f.write('{}\n'.format(value))