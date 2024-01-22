
import numpy as np
from tqdm import tqdm
import os
from input_data import load_data

if __name__ == "__main__":
    dataset = "pubmed"
    nexp = 1
    seeds = np.arange(0, nexp, dtype=int)
    model = "ARGA"

    for seed in tqdm(seeds, total=nexp):
        data = np.load("outputs/{}_{}_emb_{}.npz".format(model, dataset, seed))
        emb = data["emb"]
        _, _, _, _, _, _, labels = load_data(dataset)

        print(emb.shape, labels.shape)

        from sklearn.manifold import TSNE
        import matplotlib.pyplot as plt
        tqdm.write("plotting")
        tsne_z = TSNE(n_components=2, init="random").fit_transform(emb)
        plt.figure()
        plt.scatter(tsne_z[:, 0], tsne_z[:, 1], c=labels)
        os.makedirs("pics", exist_ok=True)
        plt.savefig("pics/{}_{}_tsne_{}.png".format(model, dataset, seed))

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