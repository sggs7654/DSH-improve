import train_and_test.function.dsh as dsh
import train_and_test.function.my_method as my
from Data.MNIST import MNIST
from Data.GeneratedData import GeneratedData
from Cluster.kmeans import Cluster
from time import time


k = 10  # 质心数目
r = 7   # 质心邻近点数目
L = 10  # 编码长度
h = 10  # eda种群数目
pn = 4  # 并联码本数
# if L > 0.5 * k * r:
#     raise RuntimeError("L>0.5kr")
data = MNIST()
# data = GeneratedData(seed=1)
start_time = time()
cluster = Cluster(data.point_set, k=k)
print("[量化耗时]", time()-start_time)

dsh.standard(data, cluster, r, L)
my.standard(data, cluster, r, L, h)
my.multiple_dict(data, cluster, r, L, h, 2)

