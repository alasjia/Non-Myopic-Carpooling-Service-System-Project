#将距离矩阵转化为时间矩阵

import pandas as pd
import numpy as np

#读取文件
path = 'E:\\BJTU\\SharedTaxi\\programC\\data_input\\network_data\\距离矩阵_0510.csv'
file_data = pd.read_csv(path, header=None, names=range(0, 225))  #header=None指没有索引，会在range范围内添加索引
#对于每个距离（m）除以速度（m/min），得到最短时间（min）
#result = file_data.apply(lambda x: round(x / 250))    #round四舍五入取整，int向0取整（绝对值偏小），一开始时间值采用整数
result = file_data.apply(lambda x: x / 250)    #改进后，时间值采用浮点数
print(result)

#保存到csv文件中
out_path = 'E:\\BJTU\\SharedTaxi\\programC\\data_input\\network_data\\时间矩阵_0601.csv'
result.to_csv(out_path, header = False, index = False, encoding = 'utf-8', float_format='%f')

