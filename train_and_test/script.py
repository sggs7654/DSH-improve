import train_and_test.function.dsh as dsh
import train_and_test.function.my_method as my
from train_and_test.function import e2lsh, pcah, klsh, elsh
from Data.MNIST import MNIST
from Data.GeneratedData import GeneratedData
from Cluster import kmeans
from Cluster import mini_batch_kmeans
from time import time


k = 40  # 质心数目
r = 5   # 质心邻近点数目
L = 7  # 编码长度, 注意'L<0.5kr'
h = 200  # eda种群数目
pn = 8  # 并联码本数

data = MNIST()
# data = GeneratedData(seed=1)
Cc = int(data.point_num * 0.03)  # Cutoff capacity截断容量在General.query中设置
for i in range(10):
    start_time = time()
    cluster1 = kmeans.Cluster(data.point_set, k=k)
    print("[kmeans量化耗时]", time()-start_time)
    print("kmeansSSE:",cluster1.inertia)
    start_time = time()
    cluster2 = mini_batch_kmeans.Cluster(data.point_set, k=k)
    print("[minibatch量化耗时]", time()-start_time)
    print("mini_batch_kmeansSSE:",cluster2.inertia)
    print("SSE Delta:",(cluster2.inertia - cluster1.inertia)/cluster2.inertia)

# dsh.standard(data, cluster, r, L, cc=Cc)
# my.multiple_dict(data, cluster, r, L, h, pn, cc=Cc)
# e2lsh.standard(data, k=L, l=500, r=2000, cc=Cc)
# elsh.standard(data, k=L, l=500, r=20, cc=Cc)
# pcah.standard(data, L=L, cc=Cc)
# klsh.standard(data, L=L, cc=Cc)

