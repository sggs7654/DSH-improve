import Screening.DSH_method
import Cluster.kmeans
import Data.GeneratedData
import numpy as np
import Screening.preprocessor

gd = Data.GeneratedData.GeneratedData()
cluster = Cluster.kmeans.Cluster(gd.point_set)
dsh = Screening.DSH_method.Storage(gd, cluster)

a = tuple(dsh.hyperplanes_dict.values())
t_list = np.empty(len(a))
for i in range(len(a)):
    t_list[i] = a[i].t
w_mat = np.empty((len(t_list), 2))  # 保存w的np矩阵, 每列为一个w向量
for i in range(len(t_list)):  # 这里的处理方式为: 先按行赋值然后转置
    w_mat[i] = a[i].w
w_mat = w_mat.transpose()
preprocessor = Screening.preprocessor.Preprocessor(gd, cluster)
print(preprocessor.t == t_list)
print(preprocessor.w == w_mat)
