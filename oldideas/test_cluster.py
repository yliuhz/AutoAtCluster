
from sklearn.cluster import KMeans, SpectralClustering, DBSCAN
import numpy as np
from sklearn.metrics import normalized_mutual_info_score as NMI, adjusted_mutual_info_score as AMI, adjusted_rand_score as ARI
from utils import make_parser
from tqdm import tqdm
import os
from community import community_louvain
from load import load_assortative, load_syn_subgraph
from utils import louvain_cluster

data2K = {
    "cora": 7,
    "citeseer": 6,
    "pubmed": 3
}


if __name__ == "__main__":
    parser = make_parser()
    args = parser.parse_args()

    dataset = args.dataset
    seeds = np.arange(0, args.nexp, dtype=int)
    K = data2K.get(dataset)
    Kmin = max(1, K-10)
    Kmax = K+50
    # adj, _, _ = load_assortative(dataset)
    adj, _, _ = load_syn_subgraph(dataset)

    results = {
        "km": {},
        "sp": {},
        # "db": {},
    }

    results_louvain = []

    for seed in tqdm(seeds, total=args.nexp):
        np.random.seed(seed)

        data = np.load("outputs/{}_emb_{}.npz".format(dataset, seed))
        emb, labels = data["emb"], data["labels"]

        # for n_clusters in range(Kmin, Kmax+1):

        #     km = KMeans(n_clusters=n_clusters, random_state=seed)
        #     sp = SpectralClustering(n_clusters=n_clusters, random_state=seed)

        #     preds_km = km.fit_predict(emb)
        #     preds_sp = sp.fit_predict(emb)

        #     nmi_km = ARI(labels, preds_km)
        #     nmi_sp = ARI(labels, preds_sp)

        #     if n_clusters not in results["km"].keys():
        #         results["km"][n_clusters] = []
        #     if n_clusters not in results["sp"].keys():
        #         results["sp"][n_clusters] = []
        #     results["km"][n_clusters].append(nmi_km)
        #     results["sp"][n_clusters].append(nmi_sp)

        preds_lo = louvain_cluster(adj, labels, random_state=seed)
        ari_lo = ARI(labels, preds_lo)
        results_louvain.append(ari_lo)

    
    os.makedirs("results", exist_ok=True)
    with open("results/results.txt", 'a+') as f:
        f.write("\n\n\n")
        # for alg, data in results.items():
        #     f.write("alg={}, k={}-{}".format(alg, Kmin, Kmax))
        #     for k, values in data.items():
        #         # f.write("{}-{}\n".format(alg, k))
        #         for d in values:
        #             f.write("{} ".format(d))
        #         f.write("\n")

        f.write("alg=louvain\n")
        for values in results_louvain:
            f.write("{} ".format(values))
        f.write("\n")



    
