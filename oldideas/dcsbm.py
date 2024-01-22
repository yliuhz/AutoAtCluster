
import numpy as np
import scipy.sparse as sp
import os


from vis import plot_superadj

class DCSBM(object):
    def __init__(self) -> None:
        pass

    def generate(self, c, w, theta, lamb, random_state=10, save=False):
        self.check(c, w, theta, lamb)
        c = np.array(c, dtype=int)
        w = np.array(w, dtype=int)
        theta = np.array(theta, dtype=float)

        K = w.shape[0]
        n = theta.shape[0]
        c_sum = np.cumsum(c)

        w_ = np.random.RandomState(seed=random_state).poisson(w)
        k = w_.sum(1)
        m = w_.sum() / 2

        w_r = np.matmul(np.transpose(k[None]), k[None])
        w_r = np.divide(w_r, (2 * m))

        w = lamb * w_ + (1 - lamb) * w_r
        w = np.array(w, dtype=int)

        w_max = int(2 * w.max())
        rand = np.random.RandomState(seed=random_state).uniform(low=0., high=1., size=(K,K,w_max))
        
        adj = np.zeros((n, n))
        labels = np.zeros(n)

        for r in range(K):
            for s in range(K):
                rand_ = rand[r, s]
                idx = 0
                
                if r == 0:
                    r_nodeid_min = 0
                else:
                    r_nodeid_min = c_sum[r-1]
                r_nodeid_max = c_sum[r]
                r_nodeids = np.arange(r_nodeid_min, r_nodeid_max)

                if s == 0:
                    s_nodeid_min = 0
                else:
                    s_nodeid_min = c_sum[s-1]
                s_nodeid_max = c_sum[s]
                s_nodeids = np.arange(s_nodeid_min, s_nodeid_max)

                r_theta = np.cumsum(theta[r_nodeids])
                s_theta = np.cumsum(theta[s_nodeids])
                r_theta[-1] = 1.0
                s_theta[-1] = 1.0
                                
                for i in range(w[r, s]):
                    d = rand_[idx]
                    idx += 1
                    p = np.searchsorted(r_theta, d)
                    p = r_nodeids[p]

                    d = rand_[idx]
                    idx += 1
                    q = np.searchsorted(s_theta, d)
                    q = s_nodeids[q]

                    adj[p, q] = adj[q, p] = 1

        
        for i in range(1, K):
            labels[c_sum[i-1]:c_sum[i]] = i

        if save:
            os.makedirs("data", exist_ok=True)
            np.savez("data/dcsbm_m.npz", adj=adj, labels=labels)
        
        return sp.coo_matrix(adj), sp.eye(adj.shape[0]), labels

    def check(self, c, w, theta, lamb):
        c = np.array(c)
        w = np.array(w)
        theta = np.array(theta)

        K = c.shape[0]
        assert w.shape[0] == K, "1, w.shape[0]={}, K={}".format(w.shape[0], K)
        n = theta.shape[0]
        assert n >= K, "2"
        assert lamb >= 0 and lamb <= 1, "3"

        assert np.all(w >= 0), "4"
        assert np.all(c > 0), "5"

        assert np.all(w == np.transpose(w)), "6"
        assert c.sum() == n, "7"


if __name__ == "__main__":
    K = 2
    n = 100
    seed=20
    p = 1.0 # wlm / wkk
    w0 = n # wkk

    c = np.ones(K, dtype=int) * (n // K)
    # w = np.eye(K, dtype=int) * n
    # w = (1 - np.eye(K, dtype=int)) * n * 2 # multi-partite
    w = [[0, 2 * n],[2 * n, n * 10]] # core-periphery
    # w = np.eye(K, dtype=int) * w0 + (1 - np.eye(K, dtype=int)) * w0 # mixing

    theta = np.random.rand(n)
    c_sum = np.cumsum(c)
    for i in range(K):
        if i == 0:
            low = 0
        else:
            low = c_sum[i-1]
        theta_ = theta[low:c_sum[i]]

        theta_ = theta_ / theta_.sum()
        theta[low:c_sum[i]] = theta_
    
    lamb = 1.0

    print('Generating graphs ...')
    dcsbm = DCSBM()
    adj, features, labels = dcsbm.generate(c, w, theta, lamb, random_state=seed, save=True)
    adj = adj.toarray()

    print('Plotting ...')
    plot_superadj(adj, K=min(adj.shape[0], 100), dataset="c_dcsbm")
    plot_superadj(1.-adj, K=min(adj.shape[0], 100), dataset="c_dcsbm_r") # reverse
    peru = np.random.RandomState(seed=seed).permutation(n)
    plot_superadj(adj[peru,:][:,peru], K=min(adj.shape[0], 100), dataset="c_dcsbm_p") # permutation
    

    print(adj.shape)
    print(labels.shape)