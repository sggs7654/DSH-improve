from General.storage import get_code
import numpy as np
from Screening.EDA import EDA
import Cluster.kmeans
import Screening.preprocessor
import General.draw
from Data.MNIST import MNIST
from Data.GeneratedData import GeneratedData

# p = [[1,0],
#      [0,1],
#      [1,1]]
# w = [[2,2],
#      [3,3]]
# t = [1,2]
# p = np.mat(p)
# w = np.mat(w)
# w = w.transpose()
# t = np.array(t)
#
# code = get_code(p, w, t)
# print(code)

mnist = MNIST()
cluster = Cluster.kmeans.Cluster(mnist.point_set)
p = Screening.preprocessor.Preprocessor(mnist, cluster)
eda = EDA(w=p.w, t=p.t, centroids=p.centroids, weight=p.weight, m=10)
eda.search()
result_indices = eda.optimum_solution[len(eda.optimum_solution) - 1]
result_indices = result_indices.astype(np.int)
w = p.w[:, result_indices]
t = p.t[result_indices]
code = get_code(mnist.point_set, w, t)
for i in code.values():
     print(len(i))

# gd = GeneratedData()
# cluster = Cluster.kmeans.Cluster(gd.point_set)
# p = Screening.preprocessor.Preprocessor(gd, cluster)
# eda = EDA(w=p.w, t=p.t, centroids=p.centroids, weight=p.weight, m=10)
# eda.search()
# draw = General.draw.draw()
# draw.new_screening(data=gd, w=p.w, t=p.t,
#                   index_screening=eda.optimum_solution[len(eda.optimum_solution)-1])
# result_indices = eda.optimum_solution[len(eda.optimum_solution) - 1]
# result_indices = result_indices.astype(np.int)
# w = p.w[:, result_indices]
# t = p.t[result_indices]
# code = get_code(gd.point_set, w, t)
# print(len(code.keys()))
