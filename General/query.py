import numpy as np

cc = 300


# point_set为数据np矩阵, 行向量
# w为超平面参数矩阵, 列向量
# t为超平面参数矩阵, 数组
# query_indices, result_indices 都是数据集中的成员变量
# bucket是storage返回的编码字典
def query(point_set, w, t, query_indices, result_indices, bucket):
    short_list_length = 0   # 临时变量, 用于累加short_list的长度
    query_set = point_set[query_indices]
    code_mat = np.sign(query_set.dot(w) - t)
    precision = np.empty(len(query_indices))
    for i in range(code_mat.shape[0]):
        code = code_mat[i].tostring()
        if code in bucket:
            return_indices = bucket[code]  # 算法返回的向量索引
        else:
            return_indices = []
        while len(return_indices) > cc:
            return_indices.pop()
        right_indices = set(return_indices).intersection(set(result_indices[i]))
        short_list_length += len(return_indices)
        precision[i] = len(right_indices) / result_indices.shape[1]
    print("short-list平均长度:", short_list_length/code_mat.shape[0])
    return np.average(precision)


# point_set为数据np矩阵, 行向量
# w为列表, 列表元素为w矩阵, 列向量
# t为列表, 列表元素为t数组
# query_indices, result_indices 都是数据集中的成员变量
# bucket是一个列表, 列表元素是编码字典
def multiple_query(point_set, w, t, query_indices, result_indices, bucket):
    short_list_length = 0   # 临时变量, 用于累加short_list的长度
    query_set = point_set[query_indices]
    pn = len(t)  # hash字典的数目
    code_mat = []  # 列表, 用于存放查询点hash编码, 列表元素为矩阵, 行向量
    for i in range(pn):
        code_mat.append(np.sign(query_set.dot(w[i]) - t[i]))
    precision = np.empty(len(query_indices))
    for i in range(len(query_indices)):  # 对每一个查询点
        return_indices = set()
        bucket_list = []
        for j in range(pn):  # 对每一个码本
            code = code_mat[j][i].tostring()  # 第j个码本中的第i行
            if code in bucket[j]:
                return_indices = return_indices.union(set(bucket[j][code]))
                bucket_list.append(set(bucket[j][code]))
        cross_retrieval_indices = cross_retrieval(return_indices, bucket_list, cc)
        right_indices = cross_retrieval_indices.intersection(set(result_indices[i]))
        short_list_length += len(cross_retrieval_indices)
        precision[i] = len(right_indices) / result_indices.shape[1]
    print("short-list平均长度:", short_list_length/len(query_indices))
    return np.average(precision)


# all是所有索引union的集合, individual是未union前的桶组成的列表
def cross_retrieval(all_index, buckets, cc):
    final_set = set()
    count_list = np.zeros(len(all_index))  # 用来统计各索引的出现次数
    indices_list = list(all_index)
    for i in range(len(indices_list)):  # 遍历所有索引
        for bucket in buckets:  # 统计索引被桶包含的次数
            if indices_list[i] in bucket:
                count_list[i] += 1
    order = np.argsort(count_list)[::-1]
    for i in range(0, min(cc, len(all_index))):
        final_set.add(indices_list[order[i]])
    return final_set
