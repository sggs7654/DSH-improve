import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from contrast.klsh import KernelLSH
from Data.GeneratedData import GeneratedData

# data = load_iris()
# X = data.data  # 二维np数组, 行向量
# y = data.target  # 一维np数组, 保存了data.data中各行向量的分类标签
# Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.1,
#                                                 random_state=42)

data = GeneratedData()
klsh = KernelLSH(nbits=8, kernel='rbf', random_state=42)  # rbf为高斯核
klsh.fit(data.point_set)
nbrs = klsh.query(data.point_set[data.query_indices], k=8)
precision = np.empty(len(data.query_indices))
for i in range(len(data.query_indices)):
    result = nbrs[i]
    right_num = len(set(result).intersection(set(data.result_indices[i])))
    precision[i] = right_num / data.result_indices.shape[1]
print(np.average(precision))
