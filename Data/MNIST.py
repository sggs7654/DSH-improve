import numpy as np
# import matplotlib.pyplot as plt
import struct


# 初始化后调用load_data(), 数据保存在mat中
class MNIST:
    path = r"D:\DATASET\mnist\t10k-images.idx3-ubyte"
    buf = None  # 保存入读缓存的数据集, 在__init__()中产生, load_data()执行后清空
    num = None  # 数据集图片数, 在__init__()中产生
    vec_length = 784  # 向量长度, 也可以通过mat.shape[1]获得
    mat = None  # num*784的np矩阵, 每行对应一个数据向量, 在load_data()中产生

    def __init__(self):  # 加载原始数据到缓冲区buf, 从文件头中读取图片数量num
        with open(self.path, 'rb') as f1:
            self.buf = f1.read()
        self.num = struct.unpack_from('>IIII', self.buf, 0)[1]

    def load_data(self):  # 把缓冲区中的原始数据转化成np矩阵
        self.mat = np.empty((self.num, 784))  # empty是创建数组最快的方法
        offset = 0
        for i in range(self.num):  # 把向量依次取出, 按行赋值到矩阵中
            offset += struct.calcsize('>IIII')
            temp = struct.unpack_from('>784B', self.buf, offset)
            row_data = np.mat(temp)
            self.mat[i] = row_data
        self.buf = None  # 矩阵加载完毕, 释放数据缓存
