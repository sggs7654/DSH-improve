import Screening.DSH_method
import General.draw
import Cluster.kmeans
import Data.GeneratedData
from Data.MNIST import MNIST

# gd = Data.GeneratedData.GeneratedData()
# cluster = Cluster.kmeans.Cluster(gd.point_set)
# dsh = Screening.DSH_method.Storage(gd, cluster)
# draw = General.draw.draw()
# # draw.hp_list(dsh.hyperplanes_dict, gd)  # 测试通过
# draw.hp_screening(dsh.hyperplanes_list, dsh.hyperplanes_dict, gd)  # 测试通过

mnist = MNIST()
mnist.load_data()
cluster = Cluster.kmeans.Cluster(mnist.point_set)
dsh = Screening.DSH_method.Storage(mnist, cluster)
print(dsh.w.shape)
print(dsh.t)
