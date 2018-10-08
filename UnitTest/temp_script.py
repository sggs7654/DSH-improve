import numpy as np
from contrast.ELSH import ELSH
from Data.GeneratedData import GeneratedData

# 手动构造数据, 进行测试
# storage 作用:
# 输入 输出:

data = GeneratedData()
# data.point_num = 5
# data.point_set = np.mat(([[0,0], [1,1], [2,2], [3,3], [4,4]]))
# print(data.point_set)

elsh = ELSH(data.point_set, k=2, l=2, r=2)
elsh.storage()
p = elsh.query(data.query_indices, data.result_indices, cc=100)
print(p)

