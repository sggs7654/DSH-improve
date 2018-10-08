import numpy as np

class ELSH:
    l = None
    k = None
    r = None
    data = None
    alpha = None  # L个正太随机np矩阵, 列向量
    beta = None  # 第一维为l, 第二维为k
    buckets = None
    axs = None
    code_mats = None
    axs_order = None

    def __init__(self, data, k, l, r):  # data为数据np矩阵, 行向量
        self.data = data
        self.k = k
        self.l = l
        self.r = r

    #  产生a矩阵, b数组, 数据hashcode表
    def storage(self):
        dim = self.data.shape[1]
        self.alpha = np.random.normal(size=(self.l, dim, self.k))
        self.buckets = [{} for x in range(self.l)]
        self.code_mats = [None for x in range(self.l)]
        self.axs = np.empty((self.l, self.data.shape[0], self.k))
        self.axs_order = np.empty((self.l, self.data.shape[0], self.k))
        for l in range(self.l):
            ax = self.data.dot(self.alpha[l])  # 编码矩阵, 行向量
            self.axs[l] = ax
            order = np.argsort(ax, axis=0)  # 对每一列, 按行排序: 从上到下, 从小到大
            self.axs_order[l] = order
            offset = int(ax.shape[0]/self.r)
            code_mat = np.empty_like(ax)
            for i in range(ax.shape[1]):  # 对每一位编码
                for row in range(ax.shape[0]):  # 对每一个数据点(按order从小到大遍历)
                    index = order[row, i]  # index为行索引
                    value = int(row/offset)
                    code_mat[index, i] = value
            self.code_mats[l] = code_mat
            for i in range(code_mat.shape[0]):  # 遍历行, 编码装箱
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
        axs = []  # 查询点ax集, 其中装了self.l个ax矩阵
        code_mat = []  # 用于存放查询点hash编码, 其格式和维度与axs完全一致
        pn = self.l  # 并行字典数(码本数)
        for i in range(pn):
            code_mat.append(np.empty((len(query_indices), self.k)))
            axs.append(query_set.dot(self.alpha[i]))
        precision = np.empty(len(query_indices))
        for i in range(len(query_indices)):  # 对每一个查询点
            return_indices = set()
            # bucket_list = []
            for j in range(pn):  # 对每一个码本
                for c in range(self.k):  # 对每一位编码
                    ax = axs[j][i, c]  # 查询点的第j个码本, 第i行第c列的ax(一个值)
                    for m in range(self.data.shape[0] - 1):  # 遍历order
                        t1 = int(self.axs_order[j][m, c])  # 根据order得到行索引
                        t2 = int(self.axs_order[j][m + 1, c])  # 根据order得到行索引
                        if self.axs[j][t1, c] < ax <= self.axs[j][t2, c]:
                            # 找到ax在数据点ax中的哪个区间内, 跟随该区间的量化编码
                            code_mat[j][i, c] = self.code_mats[j][t2, c]
                            break
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

