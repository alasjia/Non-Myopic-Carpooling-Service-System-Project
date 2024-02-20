'''
统计曼哈顿某天出租车请求到达频率，以小时为间隔绘制频率分布图，用于寻找高峰时段
'''
'''
df[]: 类似于python列表切片          
     df[0:5]——选取0、1、2、3、4行
     df['列名']——选取单列
     df[['列名1','列名2']]——选取多列
     df[df.columns[0]]——没有列名
loc: 基于position选取特定行，基于label选取特定列 
     df.loc[1]——选取第1行
     df.loc[0:5:step]——选取0、1、2、3、4、5行 
     df.loc[[0,5]]——选取特定的行，第0、5行
     df.loc[1:3, ['列名1','列名2']] or df.loc[1:3, '列名1':'列名2']——选取特定列的特定行
iloc: 基于position选取特定行/列
     行选择与loc相同
     df.loc[1:3, [1,2]] or df.loc[1:3, 1:3]——选取特定列的特定行，1:3选择1、2列
ix: 既支持label也支持position
drop: 去掉行/列   df.drop([1,2], axis = 0)——去掉1、2行

注意：在使用drop后，只能采用iloc进行[[0,5]]双括号选取特定行
'''

import pandas as pd
import matplotlib.pyplot as plt

#打开文件
data = pd.read_csv('3月1日上车时间及经纬度.csv')
#将时间字段从字符串转化为可识别时间
data['tpep_pickup_datetime'] = pd.to_datetime(data['tpep_pickup_datetime'])
#print(data['tpep_pickup_datetime'])
#获取时间中的小时部分
data['hour'] = data['tpep_pickup_datetime'].dt.hour
#print(data['hour'])

#创建横纵坐标列表
ylist = list()
xlist = list()
#统计每个时间段内请求到达数
for hour in range(24):
    num = data[data['hour'] == hour].count()[0]
    print(hour,num)
    xlist.append(hour)
    ylist.append(num)

plt.plot(xlist, ylist,'r-^')             #画散点图，*:r表示点用*表示，颜色为红色
plt.title("Statistics of request arrival number")               #设置标题
plt.xlabel("Hour")                 #横轴名称
plt.ylabel("Request Arrival Frequency")                   #纵轴名称
plt.show()                               #画图
#绘制结束，结束
data = data.drop(columns = ["hour"])