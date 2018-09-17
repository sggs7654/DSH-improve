import Screening.entropy
import numpy as np
from random import sample, randint, random


class EDA:
    h = 10  # 种群数量
    k = 0.2  # 选优系数(0.1-0.3)
    optimum_solution = None  # 保存每轮迭代中的最优解, np矩阵, 行向量
    entropy_op_solution = None  # 保存每轮迭代中的最优解对应的信息熵, 列表
    population = None  # 种群, np矩阵(h,m), 行向量
    n = None  # 超平面总数, 整数
    m = None  # 需要的超平面数量, 整数
    w = None  # 超平面参数, np矩阵, 列向量
    t = None  # 超平面参数, np数组
    centroids = None  # 质心坐标, np矩阵, 行向量
    weight = None  # 质心权重, np数组

    def __init__(self, w, t, centroids, weight, m):
        self.w, self.t, self.centroids, self.weight = w, t, centroids, weight
        self.n = len(t)
        self.m = m
        self.population = np.empty((self.h, self.m))
        for i in range(self.h):
            self.population[i] = np.array(sample(range(self.n), self.m))
        self.optimum_solution = np.zeros(self.m)
        self.entropy_op_solution = []

    def select(self):  # 选优
        entropy = np.empty(self.h)
        for row in range(self.h):  # 从种群中取出个体(按行遍历)
            w_temp = np.empty((self.w.shape[0], self.m))
            t_temp = np.empty(self.m)
            for i in range(self.m):  # 根据个体中的超平面索引, 构造计算熵需要的w, t
                w_temp[:, i] = self.w[:, int(self.population[row, i])]
                t_temp[i] = self.t[int(self.population[row,i])]
            entropy[row] = Screening.entropy.get_entropy(self.centroids,
                                                         self.weight,
                                                         w_temp,
                                                         t_temp)
        order = np.argsort(entropy)[::-1]  # 降序索引(从大到小)
        i_good = order[0:int(self.k*self.h)]
        i_bad = order[int(self.k*self.h):self.h]
        np.vstack((self.optimum_solution, self.population[order[0]]))  # 记录最优解
        self.entropy_op_solution.append(entropy[order[0]])
        return i_good, i_bad

    def fit(self, i_good, i_bad):  # 建模&采样
        model = [0 for i in range(self.m)]  # 采样概率模型, py列表, 初值为0
        for i in i_good:  # 对所有优质解
            for hp_index in self.population[i]:  # 对优质解中的每一个超平面索引
                model[hp_index] += 1  # 累加次数
        model = [x/int(self.k * self.h) for x in model]  # 除以优质解的总数, 得到采样概率
        model = [min(0.9, x) for x in model]  # 设置概率上限
        model = [max(0.1, x) for x in model]  # 设置概率下限
        for i in i_bad:
            self.population[i] = -1
            for j in range(self.m):
                while True:
                    new_hp_index = randint(0, self.n-1)
                    if random() < model[new_hp_index] and \
                            new_hp_index not in self.population[i]:
                        self.population[i, j] = new_hp_index
                        break
