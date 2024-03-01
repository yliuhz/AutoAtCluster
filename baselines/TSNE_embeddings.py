
import numpy as np
from sklearn.manifold import TSNE
import os
import random
import setproctitle

import time


def load_emb(model, dataset, seed):
    emb_paths = {
        "GAE": "/home/yliumh/github/AutoAtCluster/emb_models/GAE/outputs", 
        "VGAE": "/home/yliumh/github/AutoAtCluster/emb_models/GAE/outputs",
        "ARGA": "/home/yliumh/github/AutoAtCluster/emb_models/ARGA/ARGA/arga/outputs",
        "ARVGA": "/home/yliumh/github/AutoAtCluster/emb_models/ARGA/ARGA/arga/outputs",
        "AGE": "/home/yliumh/github/AutoAtCluster/emb_models/AGE/outputs",
        "DGI": "/home/yliumh/github/AutoAtCluster/emb_models/DGI/outputs",
        "MVGRL": "/home/yliumh/github/AutoAtCluster/emb_models/MVGRL/outputs",
        "GRACE": "/home/yliumh/github/AutoAtCluster/emb_models/GRACE/outputs",
        "GGD": "/home/yliumh/github/AutoAtCluster/emb_models/GGD/manual_version/outputs",
        "G2G": "/home/yliumh/github/graph2gauss/outputs/",
    }
    assert model in emb_paths.keys()

    GRACE_datasets = {
        "cora": "Cora",
        "citeseer": "CiteSeer", 
        "wiki": "Wiki", 
        "pubmed": "PubMed",
        "amazon-photo": "amazon-photo",
        "amazon-computers": "amazon-computers"
    }
    if model == "GRACE":
        dataset_ = GRACE_datasets[dataset]
    else:
        dataset_ = dataset

    import os
    emb_path = emb_paths[model]

    if "ogbn" in dataset:
        emb_path = os.path.join(emb_path, "{}_{}_emb1_{}.npz".format(model, dataset_, seed))
    else:
        emb_path = os.path.join(emb_path, "{}_{}_emb_{}.npz".format(model, dataset_, seed))

    data = np.load(emb_path)

    return data["emb"]

if __name__ == "__main__":
    models = ["GGD", "G2G", "AGE"] # Used embedding models

    datasets = ["cora", "citeseer", "wiki", "pubmed", "amazon-photo", "amazon-computers", "cora-full", "ogbn-arxiv"]

    for model in models:
        os.makedirs("TSNE_embs/{}".format(model), exist_ok=True)
        seeds = np.arange(3, dtype=int)

        for dataset in datasets:
            times = []
            for seed in seeds:

                if not os.path.exists("TSNE_embs/{}/{}_tsne_{}.npz".format(model, dataset, seed)):
                    print(model, dataset, seed)

                    np.random.seed(seed)
                    random.seed(seed)

                    setproctitle.setproctitle("TS-{}-{}".format(dataset[:2], seed))

                    st = time.process_time()
                    try:
                        emb = load_emb(model, dataset, seed=seed)
                    except:
                        continue
                    tsne_op = TSNE(n_components=2, random_state=seed)
                    tsne_z = tsne_op.fit_transform(emb)
                    ed = time.process_time()
                    np.savez("TSNE_embs/{}/{}_tsne_{}.npz".format(model, dataset, seed), emb=tsne_z)

                    times.append(ed-st)
                else:
                    print(f"SKIP: {model} {dataset} {seed}")
            
            with open("time.txt", "a+") as f:
                f.write("TSNE {} {}\n".format(model, dataset))
                for t in times:
                    f.write("{:.3f} ".format(t))
                f.write("\n\n")

