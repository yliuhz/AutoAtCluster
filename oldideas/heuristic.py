
import numpy as np
from vis import plot_superadj
import itertools
import matplotlib.pyplot as plt
import os

from sklearn.preprocessing import normalize, MinMaxScaler
from numpy import linalg as LA

from scipy.signal import savgol_filter
import scipy.sparse as sp

from queue import Queue
from tqdm import tqdm

# 加level级别的邻居，使图更加稠密
def bfs(adj, st, level=2, sparse=True):
    n = adj.shape[0]
    assert st >= 0 and st < n
    if sparse:
        adj = adj.toarray()
    deg = adj.sum(1)
    visit = np.zeros(n, dtype=bool)
    levels = np.zeros(n, dtype=int)
    Q = Queue(n)
    Q.put(st)
    visit[st] = True
    levels[st] = 0
    ret = np.zeros((n,n), dtype=int)
    while not Q.empty():
        u = Q.get()
        if levels[u] == level:
            if adj[st][u] == 0:
                ret[st][u] = ret[u][st] = 1
        elif levels[u] > level:
            break
        vs = np.nonzero(adj[u])[0]
        for v in vs:
            if not visit[v] and deg[v] < 1.5 * deg[u]:
                Q.put(v)
                levels[v] = levels[u] + 1
                visit[v] = True
    return ret

def bfs_sparse(adj, st, level=2):
    adj = adj.tocoo()
    n = adj.shape[0]
    assert st >= 0 and st < n
    row, col, data = adj.row, adj.col, adj.data
    # deg = adj.sum(1)
    visit = np.zeros(n, dtype=bool)
    levels = np.zeros(n, dtype=int)
    Q = Queue(n)
    Q.put(st)
    visit[st] = True
    levels[st] = 0
    st_row = adj.getrow(st)
    while not Q.empty():
        u = Q.get()
        u_row = adj.getrow(u)
        if levels[u] == level:
            if st not in u_row.indices or u not in st_row.indices:
                np.append(row, [st, u])
                np.append(col, [u, st])
                np.append(data, [1, 1])
        elif levels[u] > level:
            break
        vs = u_row.nonzero()[1]
        for v in vs:
            if not visit[v]:
                Q.put(v)
                levels[v] = levels[u] + 1
                visit[v] = True
    return sp.coo_matrix((data, (row, col)), shape=(n,n))

class Heuristic(object):

    def __init__(self) -> None:
        pass

    def detect(self, adj):
        adj = np.array(adj)
        adj = adj - np.diag(np.diag(adj)) # ensure there is no self-loop
        n = adj.shape[0]
        patient = 1
        mini_degree = 5

        deg = adj.sum(1)

        ret = []
        visit = np.zeros(n, dtype=bool)
        ids = np.arange(n)
        while True:
            u = deg.argmax(0) # TODO: not visit
            if deg[u] < mini_degree:
                break
            print('u={}'.format(u))
            c0 = [u]
            deg[u] = 0
            visit[u] = True
            score = 0
            boom = 0

            while True:
                # NOTE: how to get next walk: the number of co-neibs
                neib = (adj[u] == 1) * (visit == False)
                new_2_old = ids[neib]
                if new_2_old.shape[0] == 0:
                    break
                adj_ = adj[neib, :]
                one_common_neib = (adj_ == adj[u]).sum(1)
                v = one_common_neib.argmax()
                v = new_2_old[v]

                c0.append(v)
                deg[v] = 0
                visit[v] = True

                adj_ = adj[c0, :][:, c0]
                n_ = adj_.shape[0]
                score_ = adj_.sum() / (n_ * (n_-1) / 2)
                if score_ < score:
                    boom += 1
                    if boom >= patient:
                        break
                u = v
            ret.append(c0)
        
        return ret

    def detect2(self, adj):
        adj = sp.csr_matrix(adj)
        # adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape) # ensure there is no self-loop
        # adj.eliminate_zeros()
        n = adj.shape[0]
        # adj = (adj + sp.eye(n)).toarray()  # add self-loop
        adj = adj.toarray()


        # adj, norm = normalize(adj, return_norm=True)
        norm = LA.norm(adj, ord=2, axis=1)
        norm[norm == 0] = 1.
        norm = np.transpose(np.tile(norm, (n, 1)))
        sim = np.matmul(adj, np.transpose(adj))
        # norm1 = np.matmul(norm, np.transpose(norm))
        norm1 = norm * np.transpose(norm)
        norm2 = np.abs(norm - np.transpose(norm))
        # norm2 = (norm2 - norm2.min()) / (norm2.max() - norm2.min()) * (norm1.max() - norm1.min())
        norm = norm1 + norm2
        sim = np.divide(sim, norm)
        sim = sim - np.diag(np.diag(sim))

        # NOTE
        for i in range(n):
            # sim[i] = (sim[i] - sim[i].min()) / (sim[i].max() - sim[i].min())
            sim[i] = MinMaxScaler().fit_transform(sim[i].reshape(-1,1)).reshape(-1)
        # sim = normalize(sim, axis=1, norm="l1")

    

        # print('sim[0]: {}, {}'.format(sim[0].max(), sim[0].min()))

        plt.figure()
        plt.plot([x for x in range(n)], sim.max(1))
        plt.savefig("pics/abc.png")

        # sim = (sim - sim.min()) / (sim.max() - sim.min())
        sim = np.array(sim * 1000, dtype=int) / 1000
        # sim = np.array(sim >= 0.0, dtype=int)
        return sp.coo_matrix(sim)

    def detect3(self, adj):
        print(1)
        adj = sp.coo_matrix(adj).tocsr()
        print(2)
        adj = normalize(adj).tocoo()
        print(3)
        sim = adj.dot(adj.transpose())
        sim = sim - sp.dia_matrix((sim.diagonal()[np.newaxis, :], [0]), shape=sim.shape)

        return sim.tocoo()


    def detect4(self, adj, N=10, random_state=20):
        adj = adj.tocoo()
        n = adj.shape[0]
        row, col, data = [], [], []

        perus = []
        for i in range(n):
            i_s = adj.getrow(i).toarray().reshape(-1)
            i_n = np.nonzero(i_s)[0]
            i_peru = np.random.RandomState(seed=random_state).permutation(i_n.shape[0])
            perus.append(i_peru)   

        deg = adj.sum(1).toarray().reshape(-1)         

        for i in tqdm(range(n)):
            i_s = adj.getrow(i).toarray().reshape(-1)
            i_n = np.nonzero(i_s)[0]
            if i_n.shape[0] == 0:
                continue
            N = min(N, i_n.shape[0])
            # i_peru = np.random.RandomState(seed=random_state).permutation(i_n.shape[0])
            # i_peru = np.arange(i_n.shape[0])
            i_peru = perus[i]
            for j in range(n):
                j_s = adj.getrow(j).toarray().reshape(-1)
                j_n = np.nonzero(j_s)[0]
                if j_n.shape[0] == 0:
                    continue
                N_1 = min(N, j_n.shape[0])
                # j_peru = np.random.RandomState(seed=random_state).permutation(j_n.shape[0])
                # j_peru = np.arange(j_n.shape[0])
                j_peru = perus[j]

                i_n_idx = i_peru[:N_1]
                j_n_idx = j_peru[:N_1]
                i_n_idx = i_n[i_n_idx]
                j_n_idx = j_n[j_n_idx]

                i_m, j_m = [], []
                for ii in i_n_idx:
                    ii_s = adj.getrow(ii).toarray().reshape(-1).tolist()
                    i_m.append(ii_s)
                for jj in j_n_idx:
                    jj_s = adj.getrow(jj).toarray().reshape(-1).tolist()
                    j_m.append(jj_s)

                i_m = normalize(np.array(i_m))
                j_m = normalize(np.array(j_m))

                sim = np.matmul(i_m, np.transpose(j_m)).mean()
                
                row.append(i)
                col.append(j)
                data.append(sim)

                row.append(j)
                col.append(i)
                data.append(sim)
        sim = sp.coo_matrix((data, (row, col)), shape=(n,n))
        # sim += sp.eye(n)
        return sim

    def detect5(self, adj):
        adj = adj.tocoo()
        adj0 = adj.copy()
        n, m = adj.shape[0], adj.sum()
        for i in tqdm(range(n)):
            adj2 = bfs_sparse(adj0, st=i, level=2)
            adj = adj + adj2
        
        m2 = adj.sum()
        print("Added {} edges.".format(m2-m))

        adj = normalize(adj).tocoo()
        sim = adj.dot(adj.transpose())
        sim = sim - sp.dia_matrix((sim.diagonal()[np.newaxis, :], [0]), shape=sim.shape)

        return sim.tocoo()


    def build_adj_from_coms(self, coms, n):
        adj = np.zeros((n, n))
        for com in coms:
            inds = list(itertools.product(com, com))
            adj[list(zip(*inds))] = 1
        return adj

    def set_threshold(self, sim, adj, K=None):
        sim = np.array(sim)
        adj = np.array(adj)
        n = sim.shape[0]

        s = []
        polate = np.arange(0, 1.01, 0.01)
        for thres in polate:
            sim_ = np.array(sim >= thres, dtype=int) # 加的是1，而不是sim中本来的值
            # single = (sim_.sum(1) == 0).sum()
            sums = (sim_).sum()
            s.append(sums)
            # s.append(single)

        deg = adj.sum(1)
        s_d = []
        for d_thres in np.linspace(deg.min(), deg.max(), num=len(s)):
            deg_ = np.array(deg >= d_thres)
            single = (deg_ == 0).sum()
            s_d.append(single)

        p = [0.0]
        minus_s = n - np.array(s)
        for i in range(1, len(minus_s)):
            p.append(np.abs(minus_s[i] - minus_s[i-1]))

        p_2 = [0.0]
        for i in range(1, len(minus_s)):
            p_2.append(np.abs(p[i] - p[i-1]))

        c = []
        p1 = np.array([0.0, s[0]])
        p2 = np.array([1.0, s[-1]])
        shat = savgol_filter(s, 51, 3)
        for i, x in enumerate(polate):
            y = s[i]
            # y = shat[i]
            pt = np.array([x,y])
            c.append(np.abs(np.cross(p2 - p1, pt - p1)) / np.linalg.norm(p2 - p1))


        plt.figure()
        ax1 = plt.gca()
        ax2 = ax1.twinx()
        l1 = ax1.plot(polate, np.array(s), label="single", c="r")
        l5 = ax1.plot(polate, np.array(shat), label="single", c="purple")
        # ax1.plot(np.arange(0, 1.01, 0.01), np.array(s_d), label="degree", c="b")
        l2 = ax1.plot(polate, (p2[1] - p1[1]) / (p2[0] - p1[0]) * polate - p1[0] + p1[1], c="b")
        # l3 = ax1.plot(polate, n - np.array(s), label="cover_nodes", c="g")
        # ax2.plot(np.arange(0, 1.01, 0.01), np.array(p), label="percent", c='orange')
        # ax2.plot(np.arange(0, 1.01, 0.01), np.array(p_2), label="percent_2", c="black")
        l4 = ax2.plot(polate, np.array(c), label="distance", c="pink")
        ax1.legend(l1+l2+l4+l5, [l.get_label() for l in l1+l2+l4+l5], loc=0)
        os.makedirs("pics", exist_ok=True)
        plt.savefig("pics/covernodes_kcore.png")

        # print('mean={}'.format(sim.mean()))

        maxc = np.max(c)
        for i in range(len(c)-1, -1, -1):
            if c[i] == maxc or (i < len(c)-1 and i > 0 and c[i] >= c[i+1] and c[i] > c[i-1]):
                return polate[i]
        return polate[np.argmax(c)]

    def set_threshold_nano(self, sim, K=None):
        sim = np.array(sim).reshape(-1)
        sim = np.sort(sim)

        s = []
        polate = np.arange(0, 1.01, 0.01)
        for thres in polate:
            # sim_ = np.array(sim >= thres, dtype=int)
            # single = (sim_.sum(1) == 0).sum()
            idx = np.searchsorted(sim, thres)
            sums = (sim[idx:]).sum()
            s.append(sums)
            # s.append(single)

        c = []
        p1 = np.array([0.0, s[0]])
        p2 = np.array([1.0, s[-1]])
        for i, x in enumerate(polate):
            y = s[i]
            pt = np.array([x,y])
            c.append(np.abs(np.cross(p2 - p1, pt - p1)) / np.linalg.norm(p2 - p1))

        # print('mean={}'.format(sim.mean()))

        plt.figure()
        ax1 = plt.gca()
        ax2 = ax1.twinx()
        l1 = ax1.plot(polate, np.array(s), label="single", c="r")
        # ax1.plot(np.arange(0, 1.01, 0.01), np.array(s_d), label="degree", c="b")
        l2 = ax1.plot(polate, (p2[1] - p1[1]) / (p2[0] - p1[0]) * polate - p1[0] + p1[1], c="b")
        # l3 = ax1.plot(polate, n - np.array(s), label="cover_nodes", c="g")
        # ax2.plot(np.arange(0, 1.01, 0.01), np.array(p), label="percent", c='orange')
        # ax2.plot(np.arange(0, 1.01, 0.01), np.array(p_2), label="percent_2", c="black")
        l4 = ax2.plot(polate, np.array(c), label="distance", c="pink")
        ax1.legend(l1+l2+l4, [l.get_label() for l in l1+l2+l4], loc=0)
        os.makedirs("pics", exist_ok=True)
        plt.savefig("pics/covernodes_kcore.png")

        # maxc = np.max(c)
        # for i in range(len(c)-1, -1, -1):
        #     if c[i] == maxc or (i < len(c)-1 and i > 0 and c[i] >= c[i+1] and c[i] > c[i-1]):
        #         return polate[i]
        return polate[np.argmax(c)]

    # NOTE: 选择最大的不产生孤立点的阈值，不可行，因为初始相似度矩阵就有孤立点
    def set_threshold_3(self, sim, K=None):
        sim = sim.tocoo()
        row, col, data = sim.row, sim.col, sim.data
        idx = np.argsort(data)
        row, col, data = row[idx], col[idx], data[idx]
        n, m = sim.shape[0], data.shape[0]

        sim_ = sp.coo_matrix((np.ones(m), (row, col)), shape=(n,n))
        
        deg = np.squeeze(np.asarray(sim_.sum(1)))

        if deg.min() == 0:
            nn = np.arange(0, n)
            print(nn[deg == 0.], (nn[deg==0.]).shape[0])
            return 0.
        
        for idx, thres in enumerate(data):
            u, v = row[idx], col[idx]
            deg[u] -= 1
            if deg[u] == 0:
                return thres

        return 1.

    def set_threshold_nano_sparse(self, sim, K=None):
        print(1)
        sim = sim.tocoo()
        row, col, data = sim.row, sim.col, sim.data
        idx = np.argsort(data)
        row, col, data = row[idx], col[idx], data[idx]
        n, m = sim.shape[0], data.shape[0]

        print(2)
        thres = np.unique(data)
        # thres = np.arange(0.0, 1.001, 0.001)
        s = []
        data_s = np.cumsum(data)
        for th in tqdm(thres, total=thres.shape[0]):
            idx = np.searchsorted(data, th)
            # ss = data[idx:].sum()
            if idx == 0:
                ss = data_s[-1]
            else:
                ss = data_s[-1] - data_s[idx-1]
            s.append(ss)

        print(3)
        c = []
        p1 = np.array([0.0, s[0]])
        p2 = np.array([1.0, s[-1]])
        for i, x in tqdm(enumerate(thres), total=thres.shape[0]):
            y = s[i]
            pt = np.array([x,y])
            c.append(np.abs(np.cross(p2 - p1, pt - p1)) / np.linalg.norm(p2 - p1))

        print(4)
        plt.figure()
        ax1 = plt.gca()
        ax2 = ax1.twinx()
        l1 = ax1.plot(thres, np.array(s), label="single", c="r")
        # ax1.plot(np.arange(0, 1.01, 0.01), np.array(s_d), label="degree", c="b")
        l2 = ax1.plot(thres, (p2[1] - p1[1]) / (p2[0] - p1[0]) * thres - p1[0] + p1[1], c="b")
        # l3 = ax1.plot(polate, n - np.array(s), label="cover_nodes", c="g")
        # ax2.plot(np.arange(0, 1.01, 0.01), np.array(p), label="percent", c='orange')
        # ax2.plot(np.arange(0, 1.01, 0.01), np.array(p_2), label="percent_2", c="black")
        l4 = ax2.plot(thres, np.array(c), label="distance", c="pink")
        ax1.legend(l1+l2+l4, [l.get_label() for l in l1+l2+l4], loc=0)
        os.makedirs("pics", exist_ok=True)
        plt.savefig("pics/covernodes_kcore.png")
        
        return thres[np.argmax(c)]





    

    def normalize_adj(self, adj, thres):
        adj = adj.tocoo()
        row, col, data = adj.row, adj.col, adj.data
        n = adj.shape[0]

        data[data < thres] = 0.
        data[data >= thres] = 1.
        adj = sp.coo_matrix((data, (row, col)), shape=(n,n))
        adj.eliminate_zeros()

        return adj







if __name__ == "__main__":
    filename = "data/dcsbm_c.npz"
    data = np.load(filename)
    adj, labels = data["adj"], data["labels"]
    n = adj.shape[0]
    print('nodes={}'.format(n))

    # adj = 1 - adj
    heu = Heuristic()
    # coms = heu.detect(adj)
    # adj_ = heu.build_adj_from_coms(coms, n)
    adj_ = heu.detect2(adj)
    thres = heu.set_threshold(adj_, adj, K=5)
    adj_ = np.array(adj_ > thres, dtype=int)
    print('thres={}'.format(thres))

    plot_superadj(adj_, K=min(n, 100), dataset="heuristic")

    # for com in coms:
    #     print('com:{}, size={}'.format(com, len(com)))

    deg = adj.sum(1)
    plt.figure()
    plt.scatter([x for x in range(n)], deg)
    os.makedirs("pics", exist_ok=True)
    plt.savefig("pics/deg_ids.png")







