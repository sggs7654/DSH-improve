import train_and_test.function.dsh as dsh
import train_and_test.function.my_method as my
from train_and_test.function import e2lsh
from Data.MNIST import MNIST
from Data.GeneratedData import GeneratedData
from Cluster.kmeans import Cluster
from time import time


k = 40  # 质心数目
r = 5   # 质心邻近点数目
L = 7  # 编码长度, 注意'L<0.5kr'
h = 200  # eda种群数目
pn = 8  # 并联码本数

# data = MNIST()
data = GeneratedData(seed=1)
Cc = int(data.point_num * 0.03)  # Cutoff capacity截断容量在General.query中设置

start_time = time()
cluster = Cluster(data.point_set, k=k)
print("[量化耗时]", time()-start_time)

dsh.standard(data, cluster, r, L, cc=Cc)
my.multiple_dict(data, cluster, r, L, h, pn, cc=Cc)
e2lsh.standard(data, k=L, l=10, r=12, cc=Cc)
