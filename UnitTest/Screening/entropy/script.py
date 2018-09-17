from Screening.entropy import get_entropy
import Screening.DSH_method
import Cluster.kmeans
import Data.GeneratedData
import numpy as np

gd = Data.GeneratedData.GeneratedData()
cluster = Cluster.kmeans.Cluster(gd.point_set)
dsh = Screening.DSH_method.Storage(gd, cluster)

# t_list = np.empty(len(dsh.hyperplanes_list))
# for i in range(len(dsh.hyperplanes_list)):
#     t_list[i] = dsh.hyperplanes_list[i].t
# w_mat = np.empty((len(t_list), 2))  # 保存w的np矩阵, 每列为一个w向量
# for i in range(len(t_list)):  # 这里的处理方式为: 先按行赋值然后转置
#     w_mat[i] = dsh.hyperplanes_list[i].w
# w_mat = w_mat.transpose()
# entropy = get_entropy(cluster.centroids, dsh.weight, w_mat, t_list)
# print(entropy)

centroids = np.mat([[0, 0], [0, 1], [1, 0], [1, 1]])
# print(centroids)
weight = np.array([0.25, 0.25, 0.25, 0.25])
w_mat = np.mat([[0, -1]]).transpose()
# print(w_mat)
t_list = np.array([-0.5])
# print(t_list)
print(get_entropy(centroids, weight, w_mat, t_list) == 1)
w_mat = np.mat([[0, -1], [1, 0]]).transpose()
t_list = np.array([-0.5, 0.5])
print(get_entropy(centroids, weight, w_mat, t_list) == 2)
w_mat = np.mat([[0, -1], [1, 0]]).transpose()
t_list = np.array([-0.5, 0.5])
w_mat[:,0] = 0
t_list[0] = 0
print(get_entropy(centroids, weight, w_mat, t_list) == 1)


