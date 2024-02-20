#将QGIS输出的矩阵文件（邻接表）转换为邻接矩阵

import networkx as nx
import csv
import numpy as np

G = nx.Graph()
path = 'E:\\BJTU\\SharedTaxi\\programC\\data_input\\nyc20160301_18_19\\距离矩阵表式_0510.csv'
word_list = []
with open(path,'r',encoding='utf-8-sig') as f:
	next(f)
	for line in f:
		cols = line.strip().split(',')
		print(float(cols[2]))
		G.add_nodes_from([int(cols[0]), int(cols[1])])
		G.add_weighted_edges_from([(int(cols[0]), int(cols[1]), float(cols[2]))])


#nx.write_adjlist(G,'G.adjlist')
result = nx.to_numpy_matrix(G)
print(type(result))

np.savetxt("E:\\BJTU\\SharedTaxi\\programC\\data_input\\nyc20160301_18_19\\距离矩阵_0510.csv", result, delimiter=",",fmt='%.2f')



