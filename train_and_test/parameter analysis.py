import train_and_test.function.dsh as dsh
import train_and_test.function.my_method as my
from train_and_test.function import e2lsh, pcah, klsh, elsh
from Data.MNIST import MNIST
from Data.GeneratedData import GeneratedData
from Cluster import kmeans
from Cluster import mini_batch_kmeans
from time import time
import numpy as np
import Screening
from Screening.EDA import EDA


data = MNIST()
# data = GeneratedData(seed=3)
cluster = mini_batch_kmeans.Cluster(data.point_set)

# it=20
# k=40
# bs=160
# it_list = [2*x for x in range(1,10)]
# for delta in it_list:
#     time_list = []
#     error_list = []
#     for i in range(30):
#         start_time = time()
#         cluster = mini_batch_kmeans.Cluster(data.point_set, k=k,bs=bs,it=delta)
#         time_list.append(time()-start_time)
#         error_list.append(cluster.inertia)
#     print("[Delta]", delta)
#     print("[量化耗时]", np.average(time_list))
#     print("[量化误差]", np.average(error_list))

# delta_list = [2*x for x in range(1,10)]
# for dp in delta_list:
#     best_list = []
#     time_list = []
#     for i in range(30):
#         start_time = time()
#         p = Screening.preprocessor.Preprocessor(data, cluster)
#         eda = EDA(w=p.w, t=p.t, centroids=p.centroids, weight=p.weight, m=7, cl=dp)
#         count, best = eda.search()
#         time_list.append(time() - start_time)
#         best_list.append(best)
#     print("[delta]", dp)
#     print("[搜索用时]", np.average(time_list))
#     print("[历史最优适应度]", np.average(best_list))

r = 10   # 质心邻近点数目
L = 15  # 编码长度, 注意'L<0.5kr'
pn = 10  # 并联码本数(字典数目）
Cc = 500  # Cutoff capacity截断容量（候选集容量）
delta_list = [18]
for dp in delta_list:
    f1_list = []
    time_list = []
    for i in range(5):
        start_time = time()
        f1 = my.multiple_dict(data, cluster, r, L=L, pn=dp,cc=Cc)
        time_list.append(time() - start_time)
        f1_list.append(f1)
    print("[字典数目]", dp)
    print("[算法用时]", np.average(time_list))
    print("[F1值]", np.average(f1_list))


