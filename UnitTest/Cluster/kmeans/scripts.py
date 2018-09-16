import Cluster.kmeans
import Data.GeneratedData as data

gd = data.GeneratedData()
cluster = Cluster.kmeans.Cluster(gd.point_set)
cluster.show()  # 测试通过
