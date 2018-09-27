from General.storage import get_code
import General.draw
from General.query import query
from Screening.DSH_method import Storage
from Cluster.kmeans import Cluster
from Data.MNIST import MNIST
from Data.GeneratedData import GeneratedData
from Screening.entropy import get_entropy


def standard(data, cluster, r, L, cc):
    s = Storage(data, cluster, length=L, r=r)
    if data.point_set.shape[1] == 2:
        draw = General.draw.draw()
        draw.hp_screening(hp_list=s.hyperplanes_list, hp_dict=s.hyperplanes_dict, data=data)
    code = get_code(data.point_set, s.w, s.t)
    ap = query(data.point_set, s.w, s.t,
               data.query_indices,
               data.result_indices, code, cc)
    entropy = get_entropy(cluster.centroids, s.weight, s.w, s.t)
    print("dsh_method:", ap, "   entropy:", entropy)
    # return len(s.hyperplanes_dict)


def on_MNIST(k, r, L):
    mnist = MNIST()
    cluster = Cluster(mnist.point_set, k=k)
    s = Storage(mnist, cluster, length=L, r=r)
    code = get_code(mnist.point_set, s.w, s.t)
    ap = query(mnist.point_set, s.w, s.t,
               mnist.query_indices,
               mnist.result_indices, code)
    print("dsh_method_MNIST:",ap)


def on_GD(k, r, L):
    gd = GeneratedData()
    cluster = Cluster(gd.point_set, k=k)
    s = Storage(gd, cluster, length=L, r=r)
    draw = General.draw.draw()
    draw.hp_screening(hp_list=s.hyperplanes_list, hp_dict=s.hyperplanes_dict, data=gd)
    code = get_code(gd.point_set, s.w, s.t)
    ap = query(gd.point_set, s.w, s.t,
               gd.query_indices,
               gd.result_indices, code)
    print("平均精确度:", ap)
