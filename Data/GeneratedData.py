import numpy as np
from sklearn.neighbors import NearestNeighbors
from random import sample


class GeneratedData:
    seed = None  # 随机种子
    dim = 2  # ←跟后续代码耦合, 不建议改动
    vec_length = 2
    point_num = 200
    center_num = 7
    radius = 13
    min_coordinate = 0
    max_coordinate = 150
    point_set = None  # 一个np矩阵, 每行为一个数据向量, 行数为point_num,列数为dim
    center_set = None  # 保存中心点的np矩阵
    query_num = 100  # 查询点数
    query_indices = None  # 查询点索引, 可通过point_set[query_indices]得到查询点集
    result_indices = None  # 正确结果索引, np矩阵, 行向量

    def __init__(self, seed=1):
        self.seed = seed
        self.build_center()
        self.build_point()
        self.build_test_set()

    def build_test_set(self):
        # 这里取离查询点距离最近的前2%的点作为答案点
        # 算法会把查询点自己也算进去, 不过我们的查询点也是在数据中取的,所以不影响
        neigh = NearestNeighbors(n_neighbors=int(0.02*self.point_num),
                                 algorithm='brute')
        neigh.fit(self.point_set)
        self.query_indices = sample(range(self.point_num), self.query_num)
        self.result_indices = neigh.kneighbors(self.point_set[self.query_indices],
                                               return_distance=False)

    def reset_seed(self):  # 不用改
        self.seed = self.seed + 11
        np.random.seed(self.seed)

    #  随机取点→碰撞检测→产生中心点
    def build_center(self):
        self.center_set = np.empty((self.center_num, self.dim))
        for i in range(self.center_num):
            while True:  # 产生目标区域内的坐标点
                self.reset_seed()
                x = np.random.randint(low=self.min_coordinate + self.radius,
                                      high=self.max_coordinate - self.radius)
                self.reset_seed()
                y = np.random.randint(low=self.min_coordinate + self.radius,
                                      high=self.max_coordinate - self.radius)
                curr_vec = np.array([x,y])
                if i == 0:  # 如果这是第一个中心点,则不需要碰撞检测
                    self.center_set[i] = curr_vec
                    break
                # 对于后续的中心点,则需要检测它们是否与其他已有的中心点发生碰撞
                collision = False
                for j in range(len(self.center_set)):
                    dist = np.linalg.norm((curr_vec,self.center_set[j]))
                    if dist < 2 * self.radius:  # 碰撞定义:半径重叠
                        collision = True
                        break
                if not collision:  # 如果新点与任意一个中心点发生碰撞, 则需要重新产生新点
                    self.center_set[i] = curr_vec
                    break

    def build_point(self):
        self.point_set = np.empty((self.point_num, self.dim))
        generate_count = 0
        while True:
            if generate_count >= self.point_num:
                return
            # 随机选择一个中心点, 作为center_point
            self.reset_seed()
            center_index = np.random.randint(low=0, high=self.center_num)
            center_point = self.center_set[center_index]
            cx = center_point[0]
            cy = center_point[1]
            self.reset_seed()
            x = np.random.normal(loc=cx, scale=self.radius)
            # x = np.random.uniform(low=cx-self.radius, high=cx+self.radius)
            self.reset_seed()
            y = np.random.normal(loc=cy, scale=self.radius)
            # y = np.random.uniform(low=cy-self.radius, high=cy+self.radius)
            self.point_set[generate_count] = np.array([x,y])
            generate_count = generate_count + 1



    # def show2(self):  # 这段代码展示了如何在plt散点图中画出线段
    #     x = []
    #     y = []
    #     for i in self.center_set:
    #         x.append(i.x)
    #         y.append(i.y)
    #     plt.scatter(x, y, label='center')
    #     x.clear()
    #     y.clear()
    #     for i in self.point_set:
    #         x.append(i.x)
    #         y.append(i.y)
    #     plt.scatter(x, y, label='scatter')
    #     plt.plot([0,100], [0,100], alpha=0.2)
    #     plt.plot([0,100], [100,0], alpha=0.2)
    #     plt.plot([],[],color='#FF0000', label='a line')
    #     plt.legend()
    #     plt.show()
