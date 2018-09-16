import numpy as np


# centroids是一个行向量np矩阵
# weight是每个centroid对应的权重 (分数!前面别忘了改了!)
# w_mat是一个np矩阵,其中每列为一个w向量
# t_list是一个np数组
def get_entropy(centroids, weight, w_mat, t_list):
    regional_weight_dict = {}
    for i in range(centroids.shape[0]):
        region_code = np.sign(centroids[i].dot(w_mat) - t_list).tostring()
        if region_code in regional_weight_dict:
            regional_weight_dict[region_code] += weight[i]
        else:
            regional_weight_dict[region_code] = weight[i]
    p = np.array(tuple(regional_weight_dict.values()))
    return np.sum(-p*np.log2(p))

