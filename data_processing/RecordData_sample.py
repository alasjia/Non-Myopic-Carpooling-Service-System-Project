#对请求数据的随机抽取

#输入：从QGIS中导出的已有位置ID的请求数据，字段包括：上车时间（转为分钟）【UpTime】、上车位置ID【UpNodes_ID】、下车位置ID【OffNodes_ID】
#输出：抽取获得的请求数据，请求数量变少

import pandas as pd
import random

if __name__ == '__main__':
	#读取文件
	records = pd.read_csv(r'E:\\BJTU\\SharedTaxi\\programC\\data_input\\nyc20160301\\3月1日请求数据_转为分钟.csv', encoding = 'utf-8')
	#为达到1秒钟1个请求的水平，从每6个请求中抽取一个
	a = int(records.shape[0]/6)  #6行数据为一组需要抽取多少次
	results = pd.DataFrame()    #建立存储变量
	for i in range(a):
		c = random.sample(range(i*6, i*6+6), 1)   #按顺序每6个数字里随机选择1个
		temp = records.iloc[c]                   #对应到列表中的行
		#如果上下车站点相同，重新抽样
		while temp.iloc[0][1] == temp.iloc[0][2]:
			c = random.sample(range(i*6, i*6+6), 1)
			temp = records.iloc[c]
		results = results.append(temp, ignore_index=True)         #存储选择出的行
	#保存到csv文件中
	results.to_csv("E:\\BJTU\\SharedTaxi\\programC\\data_input\\nyc20160301\\3月1日请求数据_抽样后.csv", index = False, encoding = 'utf-8', float_format='%d')