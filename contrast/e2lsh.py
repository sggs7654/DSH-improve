import numpy as np

class E2LSH:
    l = None
    k = None
    r = None
    data = None
    alpha = None  # L个正太随机np矩阵, 列向量
    beta = None  # 第一维为l, 第二维为k
    buckets = None

    def __init__(self, data, k, l, r):  # data为数据np矩阵, 行向量
        self.data = data
        self.k = k
        self.l = l
        self.r = r

    #  产生a矩阵, b数组, 数据hashcode表
    def storage(self):
        dim = self.data.shape[1]
        self.alpha = np.random.normal(size=(self.l, dim, self.k))
        self.beta = self.r * np.random.random((self.l, self.k))
        self.buckets = [{} for x in range(self.l)]
        for l in range(self.l):
            # print(self.data.shape)
            # print(self.alpha.shape)
            # print(self.beta.shape)
            # raise RuntimeError()
            code_mat = np.floor((self.data.dot(self.alpha[l]) + self.beta[l]) / self.r)
            for i in range(self.data.shape[0]):
                code = code_mat[i].tostring()
                if code in self.buckets[l]:
                    self.buckets[l][code].append(i)
                else:
                    self.buckets[l][code] = [i]

    # query_indices查询点索引数组
    # result_indices答案点索引矩阵, 行向量
    def query(self, query_indices, result_indices, cc):
        short_list_length = 0  # 临时变量, 用于累加short_list的长度
        query_set = self.data[query_indices]
        code_mat = []  # 列表, 用于存放查询点hash编码, 列表元素为矩阵, 行向量
        pn = self.l  # 并行字典数
        for i in range(pn):
            code_mat.append(np.floor((query_set.dot(self.alpha[i]) + self.beta[i])
                                     / self.r))
        precision = np.empty(len(query_indices))
        for i in range(len(query_indices)):  # 对每一个查询点
            return_indices = set()
            bucket_list = []
            for j in range(pn):  # 对每一个码本
                code = code_mat[j][i].tostring()  # 第j个码本中的第i行
                if code in self.buckets[j]:
                    return_indices = return_indices.union(set(self.buckets[j][code]))
                    # bucket_list.append(set(self.buckets[j][code]))
            # cross_retrieval_indices = cross_retrieval(return_indices, bucket_list, cc)
            while len(return_indices) > cc:
                return_indices.pop()
            right_indices = set(return_indices).intersection(set(result_indices[i]))
            short_list_length += len(return_indices)
            precision[i] = len(right_indices) / result_indices.shape[1]
        print("short-list平均长度:", short_list_length / len(query_indices))
        return np.average(precision)

