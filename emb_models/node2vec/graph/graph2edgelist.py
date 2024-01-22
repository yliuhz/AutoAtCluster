
import os
import pickle as pkl



if __name__ == "__main__":

    datasets = [
        "cora",
        # "citeseer",
        # "pubmed"
    ]

    os.makedirs("../../data", exist_ok=True)

    for dataset in datasets:

        path = f"../../data/{dataset}/ind.{dataset.lower()}.graph"
        with open(path, 'rb') as rf:
            u = pkl._Unpickler(rf)
            u.encoding = 'latin1'
            cur_data = u.load()

        adj_list = cur_data

        with open("{}.edgelist".format(dataset), "w") as f:
            for u in range(len(adj_list)):
                for v in adj_list[u]:
                    f.write("{} {}\n".format(u, v))

    