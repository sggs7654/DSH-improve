import matplotlib.pyplot as plt
from collections import namedtuple


class draw:

    data = None
    hyperplanes_dict = None
    hyperplanes_list = None

    def __init__(self):
        ...

    def point_set(self, data):  # 接受一个维数为2的np矩阵, 每行保存一个长度为2的行向量
        if data.shape[1] > 2:
            raise RuntimeError("向量长度大于2, 无法绘制二维图像")
        x = []
        y = []
        for i in data:
            x.append(i[0])
            y.append(i[1])
        plt.scatter(x, y)
        plt.show()

    def hp_list(self, hp_dict, data):  # data为GenerateData中的PointSet类
        self.hyperplanes_dict = hp_dict
        self.data = data
        self.draw_hyperplane()

    def hp_screening(self, hp_list, hp_dict, data): # data为GenerateData中的PointSet类
        self.hyperplanes_dict = hp_dict
        self.hyperplanes_list = hp_list
        self.data = data
        self.draw_screening()

    def get_line(self, hp):
        board_min = self.data.min_coordinate - self.data.radius
        board_max = self.data.max_coordinate + self.data.radius
        if len(hp.w) > 2:
            raise RuntimeError('数据维度大于2,无法绘图')
        x = []
        y = []
        left_x = board_min
        left_y = (hp.t - hp.w[0] * left_x) / hp.w[1]
        x.append(left_x)
        y.append(left_y)
        right_x = board_max
        right_y = (hp.t - hp.w[0] * right_x) / hp.w[1]
        x.append(right_x)
        y.append(right_y)
        top_y = board_max
        top_x = (hp.t - hp.w[1] * top_y) / hp.w[0]
        x.append(top_x)
        y.append(top_y)
        bottom_y = board_min
        bottom_x = (hp.t - hp.w[1] * bottom_y) / hp.w[0]
        x.append(bottom_x)
        y.append(bottom_y)
        result_x = []
        result_y = []
        for i in range(4):
            if board_min <= x[i] <= board_max and \
                    board_min <= y[i] <= board_max:
                result_x.append(x[i])
                result_y.append(y[i])
        if len(result_x) > 2 or len(result_y) > 2:
            raise RuntimeError('不可能事件:发现超过2个点在画板内')
        return result_x,result_y

    def draw_hyperplane(self):
        for hp in self.hyperplanes_dict.values():
            px,py = self.get_line(hp)
            if max(px) - min(px) > 1000:
                raise RecursionError(hp)
            plt.plot(px, py, color='#C0C0C0', alpha=0.7)
        plt.plot([], [], color='#C0C0C0', label='candidate projection', alpha=0.7)
        x = []
        y = []
        for i in self.data.point_set:
            x.append(i[0])
            y.append(i[1])
        plt.scatter(x, y, label='data point')
        plt.legend()
        plt.show()

    def draw_screening(self):
        for hp in self.hyperplanes_dict.values():
            px,py = self.get_line(hp)
            if max(px) - min(px) > 1000:
                raise RecursionError(hp)
            plt.plot(px, py, color='#C0C0C0', alpha=0.3)
        plt.plot([], [], color='#C0C0C0', label='candidate projection', alpha=0.8)
        for hp in self.hyperplanes_list:
            px,py = self.get_line(hp)
            if max(px) - min(px) > 1000:
                raise RecursionError(hp)
            plt.plot(px, py, color='#FF0000', alpha=0.7)
        plt.plot([], [], color='#FF0000', label='high entropy projection', alpha=0.8)
        x = []
        y = []
        for i in self.data.point_set:
            x.append(i[0])
            y.append(i[1])
        plt.scatter(x, y, label='data point')
        plt.legend()
        plt.show()

    def new_screening(self, data, w, t, index_screening, centroids = None):
        hyperplane = namedtuple('hyperplane', ['w', 't'])  # 超平面命名元组
        self.data = data
        # for hp in self.hyperplanes_dict.values():
        for i in range(len(t)):
            hp = hyperplane(w=w[:, i], t=t[i])
            px, py = self.get_line(hp)
            if max(px) - min(px) > 1000:
                raise RecursionError(hp)
            plt.plot(px, py, color='#C0C0C0', alpha=0.3)
        plt.plot([], [], color='#C0C0C0', label='candidate projection', alpha=0.8)
        # for hp in self.hyperplanes_list:
        for i in index_screening:
            hp = hyperplane(w=w[:, int(i)], t=t[int(i)])
            px,py = self.get_line(hp)
            if max(px) - min(px) > 1000:
                raise RecursionError(hp)
            plt.plot(px, py, color='#FF0000', alpha=0.7)
        plt.plot([], [], color='#FF0000', label='high entropy projection', alpha=0.8)
        x = []
        y = []
        for i in self.data.point_set:
            x.append(i[0])
            y.append(i[1])
        plt.scatter(x, y, label='data point')
        if centroids is not None:
            x.clear()
            y.clear()
            for i in centroids:
                x.append(i[0])
                y.append(i[1])
            plt.scatter(x, y, label='centroids')
        plt.legend()
        plt.show()
