import numpy as np
import General.draw
from General.storage import get_code, multiple_get_code
from General.query import query, multiple_query
from Screening.EDA import EDA
from Screening.preprocessor import Preprocessor
from Cluster.kmeans import Cluster
from Data.MNIST import MNIST
from Data.GeneratedData import GeneratedData
from Screening.entropy import get_entropy
import time


def multiple_dict(data, cluster, r, L, h, pn=1):  # pn为超平面簇并联数量
    p = Preprocessor(data, cluster, r=r)
    count = 0
    w_list, t_list = [], []
    result_indices_list = []  # 里面装着set(result_indices), 用来判重
    a = time.time()
    while True:
        eda = EDA(w=p.w, t=p.t, centroids=p.centroids, weight=p.weight, m=L, h=h)
        eda.search()
        result_indices = eda.optimum_solution[len(eda.optimum_solution) - 1]
        result_indices = result_indices.astype(np.int)
        w = p.w[:, result_indices]
        t = p.t[result_indices]
        if count == 0:
            w_list.append(w)
            t_list.append(t)
            result_indices_list.append(set(result_indices))
            count += 1
        else:
            if set(result_indices) in result_indices_list:
                continue
            else:
                w_list.append(w)
                t_list.append(t)
                result_indices_list.append(set(result_indices))
                count += 1
        if count >= pn:
            break
    print("[搜索耗时]", time.time()-a)
    code = multiple_get_code(data.point_set, w_list, t_list)
    ap = multiple_query(data.point_set, w_list, t_list,
               data.query_indices,
               data.result_indices, code)
    entropy = 0
    for i in range(pn):
        entropy += get_entropy(cluster.centroids, p.weight, w_list[i], t_list[i])
    print("my_method:", ap, "   entropy:", entropy/pn)
    # return len(p.t)


def standard(data, cluster, r, L, h):
    p = Preprocessor(data, cluster, r=r)
    eda = EDA(w=p.w, t=p.t, centroids=p.centroids, weight=p.weight, m=L, h=h)
    a = time.time()
    eda.search()
    print("[搜索耗时]", time.time()-a)
    if data.point_set.shape[1] == 2:
        draw = General.draw.draw()
        draw.new_screening(data=data, w=p.w, t=p.t,
                           index_screening=eda.optimum_solution[len(eda.optimum_solution) - 1],
                           centroids=cluster.centroids)
    result_indices = eda.optimum_solution[len(eda.optimum_solution) - 1]
    result_indices = result_indices.astype(np.int)
    w = p.w[:, result_indices]
    t = p.t[result_indices]
    code = get_code(data.point_set, w, t)
    ap = query(data.point_set, w, t,
               data.query_indices,
               data.result_indices, code)
    entropy = get_entropy(cluster.centroids, p.weight, w, t)
    print("my_method:", ap, "   entropy:", entropy)
    return len(p.t)


def on_MNIST(k, r, L, h):
    time_a = time.time()
    mnist = MNIST()
    time_a = time.time() - time_a
    time_b = time.time()
    cluster = Cluster(mnist.point_set, k=k)
    time_b = time.time() - time_b
    time_c = time.time()
    p = Preprocessor(mnist, cluster, r=r)
    eda = EDA(w=p.w, t=p.t, centroids=p.centroids, weight=p.weight, m=L, h=h)
    eda.search()
    time_c = time.time() - time_c
    time_d = time.time()
    result_indices = eda.optimum_solution[len(eda.optimum_solution) - 1]
    result_indices = result_indices.astype(np.int)
    w = p.w[:, result_indices]
    t = p.t[result_indices]
    code = get_code(mnist.point_set, w, t)
    ap = query(mnist.point_set, w, t,
               mnist.query_indices,
               mnist.result_indices, code)
    time_d = time.time() - time_d
    print("[data]", time_a)
    print("[cluster]", time_b)
    print("[search]", time_c)
    print("[test]", time_d)
    print("my_method_MNIST:",ap)


def on_GD(k, r, L, h):
    mnist = GeneratedData()
    cluster = Cluster(mnist.point_set, k=k)
    p = Preprocessor(mnist, cluster, r=r)
    eda = EDA(w=p.w, t=p.t, centroids=p.centroids, weight=p.weight, m=L, h=h)
    eda.search()
    draw = General.draw.draw()
    draw.new_screening(data=mnist, w=p.w, t=p.t,
                       index_screening=eda.optimum_solution[len(eda.optimum_solution)-1])
    result_indices = eda.optimum_solution[len(eda.optimum_solution) - 1]
    result_indices = result_indices.astype(np.int)
    w = p.w[:, result_indices]
    t = p.t[result_indices]
    code = get_code(mnist.point_set, w, t)
    ap = query(mnist.point_set, w, t,
               mnist.query_indices,
               mnist.result_indices, code)
    print("my_method_GD:",ap)