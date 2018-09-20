import numpy as np


# point_set为数据np矩阵, 行向量
# w为超平面参数矩阵, 列向量
# t为超平面参数矩阵, 数组
# query_indices, result_indices 都是数据集中的成员变量
# bucket是storage返回的编码字典
def query(point_set, w, t, query_indices, result_indices, bucket):
    query_set = point_set[query_indices]
    code_mat = np.sign(query_set.dot(w) - t)
    precision = np.empty(len(query_indices))
    for i in range(code_mat.shape[0]):
        code = code_mat[i].tostring()
        if code in bucket:
            return_indices = bucket[code]  # 算法返回的向量索引
        else:
            return_indices = []
        right_indices = set(return_indices).intersection(set(result_indices[i]))
        precision[i] = len(right_indices) / result_indices.shape[1]
    return np.average(precision)