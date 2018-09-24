import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans


# 初始化时接收一个np矩阵作为数据集
class Cluster:
    k = None  # kmeans聚类中的质心数目
    point_set = None
    centroids = None
    labels = None

    def __init__(self, point_set, k=20):  # 接收一个np矩阵作为数据集
        self.point_set = point_set
        self.k = k
        estimator = KMeans(n_clusters=self.k, max_iter=3)  # 初始化聚类器
        estimator.fit(self.point_set)  # 拟合模型
        self.labels = estimator.labels_  # 获取聚类标签
        self.centroids = estimator.cluster_centers_  # 获取聚类中心

    def get_centroids_info(self):
        # 计算质心临近点索引
        nbrs = NearestNeighbors(n_neighbors=self.neighbors_size + 1, algorithm='auto').fit(self.cluster.centroids)
        if self.neighbors_size > self.cluster.centroids.shape[0]:
            raise RuntimeError("质心临近点集容量上限为'质心数目 - 1'")
        self.neighbor_indices = nbrs.kneighbors(self.cluster.centroids, return_distance=False)
        # 离当前质心最近的质心是其本身, 这不是我们所需要的,所以删去
        self.neighbor_indices = np.delete(self.neighbor_indices, 0, axis=1)  # 删除矩阵中的第0列
        # 计算质心对应簇在数据集中的占比权重
        self.weight = []
        for i in range(len(self.cluster.centroids)):
            self.weight.append(0)
        for i in range(self.point_set.point_num):
            centroids_index = self.cluster.labels[i]
            self.weight[centroids_index] += 1

    # def show(self):  # change
    #     clusters = []
    #     for i in range(self.k):
    #         clusters.append([])
    #     for i in range(self.point_set.shape[0]):
    #         clusters[self.labels[i]].append(self.point_set[i])
    #     x = []
    #     y = []
    #     for i in range(self.k):
    #         for j in clusters[i]:
    #             x.append(j[0])
    #             y.append(j[1])
    #         plt.scatter(x, y, label='Cluster ' + str(i))
    #         x.clear()
    #         y.clear()
    #     for i in range(self.k):
    #         x.append(self.centroids[i, 0])
    #         y.append(self.centroids[i, 1])
    #         # plt.scatter(self.centroids[i, 0], self.centroids[i, 1], label='Centroids' + str(i))
    #     plt.scatter(x, y, label='Centroids')
    #     plt.legend()
    #     plt.show()