from contrast.klsh.klsh import KernelLSH
import numpy as np


def standard(data, L, cc):
    klsh = KernelLSH(nbits=L, kernel='cosine',
                     epsilon=0.5)  # 几种kernel都要试一下, 结果差异很大
    klsh.fit(data.point_set)
    nbrs = klsh.query(data.point_set[data.query_indices], k=cc)  # k控制返回索引数量
    precision = np.empty(len(data.query_indices))
    for i in range(len(data.query_indices)):
        result = nbrs[i]
        right_num = len(set(result).intersection(set(data.result_indices[i])))
        precision[i] = right_num / data.result_indices.shape[1]
    ap = np.average(precision)
    print("KLSH:", ap)
