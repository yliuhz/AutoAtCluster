
import numpy as np
from load import load_assortative

class SynGraph(object):
    # adj, features: csr_matrix, labels: numpy.array
    def __init__(self, adj, features, labels, name="cora") -> None:
        self.adj = adj.toarray()
        self.features = features.toarray()
        self.labels = np.array(labels)
        self.name = name
        self.N = adj.shape[0]
        self.K = np.unique(labels).shape[0]

    # H: K*K , n=generated graph's node num
    def gen(self, H, n):
        assert H.shape[0] == H.shape[1]
        assert H.shape[0] == self.K

        


if __name__ == "__main__":
    pass