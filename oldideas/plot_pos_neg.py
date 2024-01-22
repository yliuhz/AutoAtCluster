from vis import plot_superadj
import numpy as np
from load import load_assortative, load_cora_full_im
import os
import os
from sklearn.preprocessing import normalize, MinMaxScaler, StandardScaler


if __name__ == "__main__":
    dataset = "cora-full"
    model = "MVGRL"

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

    poses = np.arange(0.01, 0.02, 0.001)
    # poses = []
    neges = np.arange(0.9, 1.0, 0.01)
    neges = []

    rates = np.arange(0.1, 1.0, 0.2)
    seeds = np.arange(0, 3, 1, dtype=int)

    rates = [0.9]
    seeds = [0]
    for rate in rates:
        for seed in seeds:
            adj, features, true_labels, mask = load_cora_full_im(rate, seed)
            
            emb_path = os.path.join(emb_paths[model], "{}_{}_emb_{:.1f}_{}.npz".format(model, dataset, rate, seed))
            data = np.load(emb_path)
            emb = data["emb"]

            emb_norm = normalize(emb)
            f_adj = np.matmul(emb_norm, np.transpose(emb_norm)) # sim
            cosine = f_adj
            cosine = cosine.reshape([-1,])
            
            # pos_inds = np.argpartition(-cosine, pos_num)[:pos_num]
            # neg_inds = np.argpartition(cosine, neg_num)[:neg_num]

            for pos in poses:
                pos_num = round(emb.shape[0] * emb.shape[0] * pos)
                pos_inds = np.argpartition(-cosine, pos_num)[:pos_num]
                adj = np.zeros((emb.shape[0], emb.shape[0]))
                xinds = pos_inds // emb.shape[0]
                yinds = pos_inds % emb.shape[0]
                for x, y in zip(xinds, yinds):
                    adj[x][y] = 1.
                plot_superadj(adj, K=100, sparse=False, labels=true_labels, vline=True, dataset="pos_{:.1f}_{:.5f}_{:d}".format(rate, pos, seed), folder="cora-full-im/pos")

            for neg in neges:
                print(f"neg={neg:.2f}")
                neg_num = round(emb.shape[0] * emb.shape[0] * neg)
                neg_inds = np.argpartition(cosine, neg_num)[:neg_num]
                adj = np.zeros((emb.shape[0], emb.shape[0]))
                xinds = neg_inds // emb.shape[0]
                yinds = neg_inds % emb.shape[0]
                for x, y in zip(xinds, yinds):
                    adj[x][y] = 1.
                plot_superadj(adj, K=100, sparse=False, labels=true_labels, vline=True, dataset="neg_{:.1f}_{:.5f}_{:d}".format(rate, neg, seed), folder="cora-full-im/neg")
