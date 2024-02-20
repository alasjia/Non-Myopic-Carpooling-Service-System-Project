#实验：将数组变为矩阵
import numpy as np
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
A=np.array([[1,2],
            [3,4],
            [5,6],
            [7,8]])
# A是一个向量矩阵：euclidean代表欧式距离
distA = pdist(A,metric='euclidean')
# 将distA数组变成一个矩阵
distB = squareform(distA)
print(distB)