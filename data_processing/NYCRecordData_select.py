#NYC Record Data的初步处理，用于提取输入QGIS的GPS点
#一方面用于站点选择，一方面用于请求数据中上下车位置的站点匹配
#输入：从TLC官网下载的某月请求数据
#输出：特定时间段内的、仅具有所需字段的请求数据

import pandas as pd
import numpy as np
import datetime as dt

if __name__ == '__main__':
	#读取文件
	data_ = pd.read_csv(r'E:\\BJTU\\SharedTaxi\\NYCdata\\201603\\yellow_tripdata_2016-03.csv', encoding = 'utf-8')
	'''
	#【字段筛选】
	#保留上车时间和经纬度的列
	temp = data_.iloc[:, 1:7]
	temp = temp.drop(['tpep_dropoff_datetime', 'passenger_count', 'trip_distance'], axis = 1)
	print("16年3月请求总数为%d" % temp.shape[0])

	#【数据清洗，还需改进】
	#删除经纬度中的缺失值0
	temp = temp.replace(0, np.NaN)
	num_null = temp.isnull().sum()
	print(num_null)
	temp = temp.dropna(how = 'any')
	print("去除缺失值后16年3月请求总数为%d" % temp.shape[0])

	#【获取指定日期范围内数据】
	temp["tpep_pickup_datetime"] = pd.to_datetime(temp["tpep_pickup_datetime"])    #将时间字段从字符串转换为可识别的时间
	#print(temp["tpep_pickup_datetime"])
	#截取数据中3月1日的部分
	s_date = dt.datetime.strptime('2016-03-01 0:0:0', '%Y-%m-%d %H:%M:%S').date()
	e_date = dt.datetime.strptime('2016-03-02 0:0:0', '%Y-%m-%d %H:%M:%S').date()
	temp = temp[(temp['tpep_pickup_datetime'].dt.date >= s_date) & (temp['tpep_pickup_datetime'].dt.date < e_date)]
	print("3月1日纽约市黄色出租车请求数量为%d" % temp.shape[0])
	#获取时间中的小时部分
	temp["hour"] = temp["tpep_pickup_datetime"].dt.hour
	#筛选时间,晚18点至19点
	temp1 = temp[(17 < temp["hour"]) & (temp["hour"] < 19)]
	temp1 = temp1.drop(columns = ["hour"])
	temp1.to_csv("E:\\BJTU\\SharedTaxi\\python\\三月一日18至19请求时间及经纬度.csv", index = False, encoding = 'utf-8')
	'''

	#【字段筛选】
	#保留上、下车时间和经纬度的列
	temp = data_.iloc[:, 1:11]
	temp = temp.drop(['passenger_count', 'trip_distance', 'RatecodeID', 'store_and_fwd_flag'], axis = 1)
	print(temp.shape)

	#【数据清洗，还需改进】
	#删除经纬度中的缺失值0
	temp = temp.replace(0, np.NaN)
	num_null = temp.isnull().sum()
	print(num_null)
	temp = temp.dropna(how = 'any')
	print(temp.shape)

	#【获取指定日期范围内数据】
	temp["tpep_pickup_datetime"] = pd.to_datetime(temp["tpep_pickup_datetime"])    #将时间字段从字符串转换为可识别的时间
	#print(temp["tpep_pickup_datetime"])
	#截取数据中3月1日的部分
	s_date = dt.datetime.strptime('2016-03-01 0:0:0', '%Y-%m-%d %H:%M:%S').date()
	e_date = dt.datetime.strptime('2016-03-02 0:0:0', '%Y-%m-%d %H:%M:%S').date()
	temp = temp[(temp['tpep_pickup_datetime'].dt.date >= s_date) & (temp['tpep_pickup_datetime'].dt.date < e_date)]
	print("3月1日纽约市黄色出租车请求数量为%d" % temp.shape[0])
	#temp.to_csv("E:\\BJTU\\SharedTaxi\\python\\3月1日上下车时间及经纬度.csv", index = False, encoding = 'utf-8')
	
	#获取时间中的小时部分
	temp["hour"] = temp["tpep_pickup_datetime"].dt.hour
	#筛选时间,晚18点至19点
	temp1 = temp[(17 < temp["hour"]) & (temp["hour"] < 19)]
	temp1 = temp1.drop(columns = ["hour"])
	temp1.to_csv("E:\\BJTU\\SharedTaxi\\python\\3月1日18时至19时请求时间及经纬度.csv", index = False, encoding = 'utf-8')