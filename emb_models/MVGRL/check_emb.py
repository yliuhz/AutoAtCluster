
import numpy as np
from tqdm import tqdm
import os

if __name__ == "__main__":
    dataset = "cora"
    nexp = 1
    seeds = np.arange(0, nexp, dtype=int)

    for seed in tqdm(seeds, total=nexp):
        data = np.load("outputs/MVGRL_{}_emb_{}.npz".format(dataset, seed))
        emb, labels = data["emb"], data["labels"]

        from sklearn.manifold import TSNE
        import matplotlib.pyplot as plt
        tqdm.write("plotting")
        tsne_z = TSNE(n_components=2, init="random").fit_transform(emb)
        plt.figure()
        plt.scatter(tsne_z[:, 0], tsne_z[:, 1], c=labels)
        os.makedirs("pics", exist_ok=True)
        plt.savefig("pics/MVGRL_{}_tsne_{}.png".format(dataset, seed))

        from sklearn.cluster import KMeans
        from sklearn.metrics import normalized_mutual_info_score as NMI, adjusted_mutual_info_score as AMI, adjusted_rand_score as ARI
        tqdm.write("clustering")
        clustering = KMeans(n_clusters=7)
        preds = clustering.fit_predict(emb)
        nmi = NMI(labels, preds)
        ami = AMI(labels, preds)
        ari = ARI(labels, preds)
        os.makedirs("results", exist_ok=True)
        with open("results/results.txt", "a+") as f:
            f.write("{} {} {}\n".format(nmi, ami, ari))