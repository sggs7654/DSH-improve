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


def multiple_get_code(point_set, w, t):
    if len(w) != len(t):
        raise RuntimeError("w,t列表长度不同")
    pn = len(t)
    bucket = [{} for x in range(pn)]
    for j in range(pn):
        code_mat = np.sign(point_set.dot(w[j]) - t[j])
        for i in range(point_set.shape[0]):
            code = code_mat[i].tostring()
            if code in bucket[j]:
                bucket[j][code].append(i)
            else:
                bucket[j][code] = [i]
    # for i in bucket:
    #     print(len(i.keys()))
    return bucket
