import numpy as np


# point_set为数据np矩阵, 行向量
# w为超平面参数矩阵, 列向量
# t为超平面参数矩阵, 数组
def get_code(point_set, w, t):
    bucket = {}
    code_mat = np.sign(point_set.dot(w) - t)  # That's so cool!
    for i in range(point_set.shape[0]):
        code = code_mat[i].tostring()
        if code in bucket:
            bucket[code].append(i)
        else:
            bucket[code] = [i]
    return bucket
