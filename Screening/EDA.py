import Screening.entropy
import numpy as np
from random import sample, randint, random


class EDA:
    h = 10  # 种群数量
    k = 0.2  # 选优系数(0.1-0.3)
    convergence_limit = 5  # 收敛限制: 连续多少次结果无改善时停止搜索
    optimum_solution = None  # 保存每轮迭代中的最优解, np矩阵, 行向量
    op_entropy_list = None  # 保存每轮迭代中的最优解对应的信息熵, 列表
    ave_entropy_list = None  # 保存每轮迭代中的种群平均的信息熵, 列表
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
        self.op_entropy_list = []
        self.ave_entropy_list = []

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
        # 记录最优解及其对应的熵
        if self.optimum_solution is None:
            self.optimum_solution = self.population[order[0]]
        else:
            self.optimum_solution = np.vstack((self.optimum_solution,
                                               self.population[order[0]]))
        self.op_entropy_list.append(entropy[order[0]])
        self.ave_entropy_list.append(np.average(entropy))
        return i_good, i_bad

    def fit(self, i_good, i_bad):  # 建模&采样
        model = [0 for i in range(self.n)]  # 采样概率模型, py列表, 初值为0
        for i in i_good:  # 对所有优质解
            for hp_index in self.population[i]:  # 对优质解中的每一个超平面索引
                model[int(hp_index)] += 1  # 累加次数
        model = [x/int(self.k * self.h) for x in model]  # 除以优质解的总数, 得到采样概率
        model = [min(0.9, x) for x in model]  # 设置采样概率上限
        model = [max(0.1, x) for x in model]  # 设置采样概率下限
        for i in i_bad:  # 对所有劣质解
            self.population[i] = -1  # 先清除原有数据
            for j in range(self.m):
                while True:
                    new_hp_index = randint(0, self.n-1)  # 随机取一个超平面
                    # random()产生一个0-1之间的随机小数
                    if random() < model[new_hp_index] and \
                            new_hp_index not in self.population[i]:
                        self.population[i, j] = new_hp_index
                        break

    def search(self):
        count = 0  # 迭代计数器
        convergence_count = 0  # 收敛计数器
        while True:
            ig, ib = self.select()
            self.fit(ig, ib)
            print("[count]", count,
                  "   [best_entropy]", self.op_entropy_list[count],
                  "   [ave_entropy]", self.ave_entropy_list[count])
            if count > 0:
                if self.op_entropy_list[count] <= self.op_entropy_list[count - 1]:
                    convergence_count += 1  # 结果无改善, 收敛计数器加一
                else:
                    convergence_count = 0  # 结果改善, 收敛计数器清零
            count += 1
            if convergence_count > self.convergence_limit:  # 连续多次结果无改善, 则退出循环
                break
