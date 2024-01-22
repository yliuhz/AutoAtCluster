
import os
import pickle as pkl

import numpy as np
import scipy.sparse as sp

if __name__ == "__main__":

    # datasets = [
    #     "cora",
    #     "citeseer",
    #     "pubmed"
    # ]

    # for dataset in datasets:
    #     path = "ind.{}.graph".format(dataset)
    #     with open(path, 'rb') as rf:
    #         u = pkl._Unpickler(rf)
    #         u.encoding = 'latin1'
    #         cur_data = u.load()

    #     adj_list = cur_data

    #     with open("{}.edgelist".format(dataset), "w") as f:
    #         for u in range(len(adj_list)):
    #             for v in adj_list[u]:
    #                 f.write("{} {}\n".format(u, v))

    datasets = [
        "amazon-photo",
        "amazon-computers"
    ]

    map2names = {
        "amazon-photo": "/data/liuyue/New/SBM/mySBM/data/amazon_electronics_photo.npz",
        "amazon-computers": "/data/liuyue/New/SBM/mySBM/data/amazon_electronics_computers.npz",
    }

    for dataset in datasets:
        data = np.load(map2names[dataset])
        # print(list(data.keys()))
        adj_data, adj_indices, adj_indptr, adj_shape = data["adj_data"], data["adj_indices"], data["adj_indptr"], data["adj_shape"]
        # attr_data, attr_indices, attr_indptr, attr_shape = data["attr_data"], data["attr_indices"], data["attr_indptr"], data["attr_shape"]
        # labels = data["labels"]

        adj = sp.csr_matrix((adj_data, adj_indices, adj_indptr), shape=adj_shape).tocoo()

        edge_index = adj.nonzero()

        with open("{}.edgelist".format(dataset), "w") as f:
            for u, v in zip(*edge_index):
                f.write("{} {}\n".format(u, v))


    