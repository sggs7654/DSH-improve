import numpy as np
from sklearn.neighbors import NearestNeighbors
from random import sample
import struct


class MNIST:
    path = r"D:\DATASET\mnist\t10k-images.idx3-ubyte"
    buf = None  # 保存入读缓存的数据集, 在__init__()中产生, load_data()执行后清空
    point_num = None  # 数据点数
    vec_length = 784  # 向量长度, 也可以通过mat.shape[1]获得
    point_set = None  # np矩阵, 行向量, 在load_data()中生成
    query_num = 100  # 查询点数
    query_indices = None  # 查询点索引, 可通过point_set[query_indices]得到查询点集
    result_indices = None  # 正确结果索引, np矩阵, 行向量

    def __init__(self):  # 加载原始数据到缓冲区buf, 从文件头中读取图片数量num
        with open(self.path, 'rb') as f1:
            self.buf = f1.read()
        self.point_num = struct.unpack_from('>IIII', self.buf, 0)[1]
        self.load_data()
        self.build_test_set()

    def load_data(self):  # 把缓冲区中的原始数据转化成np矩阵
        self.point_set = np.empty((self.point_num, 784))  # empty是创建数组最快的方法
        offset = 0
        for i in range(self.point_num):  # 把向量依次取出, 按行赋值到矩阵中
            offset += struct.calcsize('>IIII')
            temp = struct.unpack_from('>784B', self.buf, offset)
            row_data = np.mat(temp)
            self.point_set[i] = row_data
        self.buf = None  # 矩阵加载完毕, 释放数据缓存

    def build_test_set(self):
        # 这里取离查询点距离最近的前2%的点作为答案点
        # 算法会把查询点自己也算进去, 不过我们的查询点也是在数据中取的,所以不影响
        neigh = NearestNeighbors(n_neighbors=int(0.02*self.point_num),
                                 algorithm='brute')
        neigh.fit(self.point_set)
        self.query_indices = sample(range(self.point_num), self.query_num)
        self.result_indices = neigh.kneighbors(self.point_set[self.query_indices],
                                               return_distance=False)
