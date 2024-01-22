
import numpy as np
from load import load_assortative

''' Sample subgraphs in large graphs '''
def sample_subgraph(dataset="ogbn-arxiv", rate=0.1, target_nodesize=None):
    assert rate is None or target_nodesize is None, "Either 'rate' or 'target_nodesize' must be None!"

    adj, features, labels = load_assortative(dataset)
    nclass = np.unique(labels).shape[0]
    

