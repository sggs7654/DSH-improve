import numpy as np
from sklearn.decomposition import PCA


# data是数据np矩阵, 行向量
# L是编码长度, 决定了PCAH返回几个超平面
def PCAH(data, L):
    center = np.average(data, axis=0)
    pca = PCA(n_components=L)
    pca.fit(data)
    w = pca.components_
    t = np.empty(L)
    for i in range(L):
        t[i] = w[i].dot(center)
    w = w.transpose()
    return w, t
