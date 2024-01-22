
import numpy as np
import matplotlib.pyplot as plt
import os

class UnionFind(object):

    def __init__(self, n):
        self.arr = np.ones(n, dtype=int) * (-1)

    def find(self, x):
        f = self.arr[x]
        while f >= 0:
            x = f
            f = self.arr[x]
        return x
    
    def union(self, x, y):
        f1 = self.find(x)
        f2 = self.find(y)

        if f1 != f2:
            n1 = -self.arr[f1]
            n2 = -self.arr[f2]
            if n1 > n2:
                self.arr[f1] -= n2
                self.arr[f2] = f1
            else:
                self.arr[f2] -= n1
                self.arr[f1] = f2

class CoreDecomposition(object):

    def __init__(self):
        pass

    def decomposition(self, adj):
        adj = np.array(adj)
        n = adj.shape[0]
        adj -= np.diag(np.diag(adj)) # make sure there's no self-loop

        d = np.array(adj.sum(1), dtype=int) # degree
        D = {k:[] for k in range(n)}
        c = np.zeros(n, dtype=int)


        for i in range(n):
            D[d[i]].append(i)

        # print("#"*5, "DEBUG", "#"*5)
        # print(d)
        # print(D)

        for k in range(n):
            for u in D[k]:
                c[u] = k
                for v in range(n):
                    if adj[u][v] == 1 and d[v] > k:
                        D[d[v]].remove(v)
                        D[d[v]-1].append(v)
                        d[v] -= 1

        return c    

    def cover_nodes(self, c):
        c = np.array(c)
        n = c.shape[0]

        ret = np.zeros(n)
        ret[0] = n

        cnt = np.zeros(n)
        for i in range(n):
            cnt[c[i]] += 1

        for i in range(1, n):
            ret[i] = ret[i-1] - cnt[i-1]

        return ret

    def connect_components(self, adj, c):
        adj = np.array(adj)
        c = np.array(c)
        n = c.shape[0]
        cc_ret = np.zeros(n)
        sing_ret = np.zeros(n)

        cat = {k:[] for k in range(n)} # k-core-->[nodeids]
        for i in range(n):
            cat[c[i]].append(i)

        uf = UnionFind(n)
        nodes = []
        for k in range(n-1,-1,-1):
            nodes += cat[k]
            adj_ = adj[nodes,:][:,nodes]
            st, ed = np.nonzero(adj_)

            for u, v in zip(st, ed):
                uf.union(u, v)
            
            cc_ret[k] = (uf.arr < -1).sum()
            sing_ret[k] = (uf.arr == -1).sum()

        return cc_ret, sing_ret




if __name__ == "__main__":
    filename = "data/dcsbm_d.npz"
    data = np.load(filename)
    adj, labels = data["adj"], data["labels"]
    n = adj.shape[0]
    print('nodes={}'.format(n))

    adj = 1. - adj
    cd = CoreDecomposition()
    c = cd.decomposition(adj)
    cover_nodes = cd.cover_nodes(c)
    connect_components, single_nodes = cd.connect_components(adj, c)

    plt.figure()
    n_ = 20
    plt.plot([k for k in range(n_)], cover_nodes[:n_], 'b', label="cover")
    plt.plot([k for k in range(n_)], connect_components[:n_], 'r', label="connect")
    plt.plot([k for k in range(n_)], single_nodes[:n_], 'g', label="single")
    plt.legend()
    os.makedirs("pics", exist_ok=True)
    plt.savefig("pics/covernodes_kcore.png")

    print('max(cc)={}'.format(connect_components.max()))






