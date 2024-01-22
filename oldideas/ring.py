
import numpy as np
from sklearn.preprocessing import normalize


if __name__ == "__main__":

    n = 10
    ret = np.zeros((n,n))

    for i in range(n-1):
        ret[i][i-1] = 1
        ret[i][i+1] = 1
    ret[n-1][0] = 1
    ret[n-1][n-2] = 1

    print('origin')
    # print(ret)
    adj = ret.copy()
    adj = adj - np.diag(np.diag(adj))
    adj = adj + np.eye(n)
    adj = normalize(adj)
    sim = np.matmul(adj, np.transpose(adj))
    print(sim)


    ret_2 = np.matmul(ret, ret)
    ret = ret + ret_2
    ret[ret > 1] = 1
    print("ret_1 + ret_2")
    # print(ret)
    adj = ret.copy()
    adj = adj - np.diag(np.diag(adj))
    adj = adj + np.eye(n)
    adj = normalize(adj)
    sim = np.matmul(adj, np.transpose(adj))
    print(sim)
