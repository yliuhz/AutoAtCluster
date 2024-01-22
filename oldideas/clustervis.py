
from sklearn.manifold import TSNE
# from cuml.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
from utils import make_parser
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)

if __name__ == "__main__":
    # np.random.seed(20)

    parser = make_parser()
    args = parser.parse_args()
    dataset = args.dataset
    
    # dataset = "genius"
    # dataset = "cora"
    # dataset = "fb100-penn94"
    # dataset = "citeseer"
    # dataset = "ogbn-arxiv"

    seeds = np.arange(0, args.nexp, dtype=int)
    # seeds = [11]
    for seed in seeds:
        np.random.seed(seed)

        data = np.load("outputs/{}_emb_{}.npz".format(dataset, seed))
        emb = data["emb"]
        labels = data["labels"]

        print('Plotting ...')
        tsne_z = TSNE(n_components=2, init="random", random_state=seed).fit_transform(emb)
        plt.figure(figsize=(10,10))
        plt.scatter(tsne_z[:,0], tsne_z[:,1], c=labels)
        plt.savefig("pics/cluster_{}_{}.png".format(dataset, seed))

    