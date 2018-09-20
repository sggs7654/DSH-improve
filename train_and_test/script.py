import train_and_test.function.dsh as dsh
import train_and_test.function.my_method as my
from Data.MNIST import MNIST
from Data.GeneratedData import GeneratedData
from Cluster.kmeans import Cluster

k = 10
r = 3
L = 10
h = 100
# if L > 0.5 * k * r:
#     raise RuntimeError("L>0.5kr")

# dsh.on_MNIST(k, r, L)
# dsh.on_GD(k, r, L)
# my.on_MNIST(k, r, L, h)
# my.on_GD(k, r, L, h)

# data = MNIST()
data = GeneratedData()
cluster = Cluster(data.point_set, k=k)
a = dsh.standard(data, cluster, r, L)
b = my.standard(data, cluster, r, L, h)
print(a,b)