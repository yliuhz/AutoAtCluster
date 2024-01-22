
from load import load_assortative
from vis import plot_superadj

import numpy as np

if __name__ == "__main__":

    dataset = "cora"
    K = 7
    adj, features, labels = load_assortative(dataset=dataset)
    seed = 1234

    adj = adj.toarray()
    # idx = np.argsort(labels)
    deg = adj.sum(1)
    deg_s = np.argsort(-deg)
    
    nclass = []
    for i in range(K):
        print("Class {}: {}".format(i, (labels == i).sum()))
        nclass.append((labels == i).sum())

    assert sum(nclass) == adj.shape[0]

    N = 20
    c1 = 1
    c2 = 2
    
    # idx_1 = np.random.RandomState(seed).permutation(nclass[0])[:N//2]
    # idx_2 = np.random.RandomState(seed).permutation(nclass[1])[:N//2]
    deg_1 = adj[(labels == c1), :][:, (labels == c1)].sum(1)
    deg_2 = adj[(labels == c2), :][:, (labels == c2)].sum(1)
    deg_1_s = np.argsort(-deg_1)
    deg_2_s = np.argsort(-deg_2)
    idx_1 = deg_1_s[:N//2]
    idx_2 = deg_2_s[:N//2]

    nn = np.arange(adj.shape[0])
    idx_1 = nn[(labels == c1)][idx_1]
    idx_2 = nn[(labels == c2)][idx_2]

    idx = np.hstack([idx_1, idx_2])

    adj = adj[idx, :][:, idx]
    plot_superadj(adj, K=adj.shape[0], sparse=False, labels=labels[idx], dataset="{}_subgraph".format(dataset), vline=True)
    features = features.toarray()[idx]
    labels = labels[idx]

    import os
    os.makedirs("data/{}_subgraph".format(dataset), exist_ok=True)
    np.savez("data/{}_subgraph/graph.npz".format(dataset), adj=adj, features=features, labels=labels)