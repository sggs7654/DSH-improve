from sklearn.neighbors import NearestNeighbors
import numpy as np


class Preprocessor:

    neighbors_size = 4  # 质心临近点集容量(其上限为质心数目 - 1), 其值最好大于length
    point_set = None
    cluster = None
    neighbor_indices = None  # 质心临近点索引矩阵, 第一维为质心索引,第二维为临近点索引
    weight = None  # 质心对应簇在数据集中的占比权重所组成的列表,其索引与质心索引一致
    centroids = None
    w = None  # np矩阵, 每列为一个w向量
    t = None  # np数组

    def __init__(self,point_set, cluster):
        self.point_set = point_set
        self.cluster = cluster
        self.centroids = cluster.centroids
        self.get_centroids_info()  # 计算: 1.质心最近邻索引, 2.各簇占比权重
        self.get_hyperplane_set()  # 计算相邻质心间的超平面参数(w,t)

    def get_centroids_info(self):
        # 计算质心临近点索引
        nbrs = NearestNeighbors(n_neighbors=self.neighbors_size + 1, algorithm='auto').fit(self.cluster.centroids)
        if self.neighbors_size > self.cluster.centroids.shape[0]:
            raise RuntimeError("质心临近点集容量上限为'质心数目 - 1'")
        self.neighbor_indices = nbrs.kneighbors(self.cluster.centroids, return_distance=False)
        # 离当前质心最近的质心是其本身, 这不是我们所需要的,所以删去
        self.neighbor_indices = np.delete(self.neighbor_indices, 0, axis=1)  # 删除矩阵中的第0列
        # 计算质心对应簇在数据集中的占比权重
        self.weight = np.zeros(len(self.cluster.centroids))
        for i in range(self.point_set.point_num):
            centroids_index = self.cluster.labels[i]
            self.weight[centroids_index] += 1
        self.weight = self.weight / self.point_set.point_num

    def get_hyperplane_set(self):
        hyperplanes_dict = {}
        for i in range(0, len(self.cluster.centroids)):  # 遍历质心索引
            for j in range(self.neighbors_size):  # 遍历邻近质心索引
                centroids_index1 = i
                centroids_index2 = self.neighbor_indices[i, j]
                # 利用'集合→元组'的方式处理key, 避免索引交换顺序后被重复计算
                key = tuple({centroids_index1, centroids_index2})
                if key not in hyperplanes_dict.keys():
                    u1 = self.cluster.centroids[centroids_index1]
                    u2 = self.cluster.centroids[centroids_index2]
                    w = u1 - u2
                    t = np.dot((u1 + u2)/2, w)  # 点积运算
                    hyperplanes_dict[key] = (w, t)
        hyperplanes = tuple(hyperplanes_dict.values())
        self.t = np.empty(len(hyperplanes_dict))
        for i in range(len(hyperplanes_dict)):
            self.t[i] = hyperplanes[i][1]
        self.w = np.empty((len(self.t), self.point_set.dim))  # 保存w的np矩阵, 每列为一个w向量
        for i in range(len(self.t)):  # 这里的处理方式为: 先按行赋值然后转置
            self.w[i] = hyperplanes[i][0]
        self.w = self.w.transpose()
