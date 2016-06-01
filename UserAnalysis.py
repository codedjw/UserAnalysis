#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import offsetbox
from sklearn import (manifold, datasets, decomposition, ensemble,
                     discriminant_analysis, random_projection)
from sklearn.cluster import KMeans, DBSCAN
import pandas as pd
# 数据库访问
import MySQLdb, datetime

import sys,os
#获取脚本文件的当前路径
def cur_file_dir():
     #获取脚本路径
     path = sys.path[0]
     #判断为脚本文件还是py2exe编译后的文件，如果是脚本文件，则返回的是脚本的目录，如果是py2exe编译后的文件，则返回的是编译后的文件路径
     if os.path.isdir(path):
         return path
     elif os.path.isfile(path):
         return os.path.dirname(path)
#打印结果

cur_file_dir = cur_file_dir()

## 读取数据（user_dim.txt）
X = np.loadtxt('/Users/dujiawei/git/UserAnalysis/result/user_dim.txt')
# Open database connection
conn = MySQLdb.connect(host='localhost',user='root',passwd='',db='qyw', charset='utf8')
c = ['yellowgreen', 'lightskyblue', 'lightcoral', 'gold', 'red']

## t-SNE算法降维+DBSCAN聚类（效果最好）
# ###############################################################################
# not good
# Compute Kmeans
# # Visualize the results on PCA-reduced data
# tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
# reduced_data = tsne.fit_transform(X)
# kmeans = KMeans(init='k-means++', n_clusters=100, n_init=1)
# cls = kmeans.fit(reduced_data)
# print cls.cluster_centers_
# print cls.labels_
# print cls.inertia_

##############################################################################
# Visualize the results on PCA-reduced data
tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
reduced_data = tsne.fit_transform(X)
plt.scatter(reduced_data[:, 0], reduced_data[:, 1])
plt.savefig(cur_file_dir+'/result/'+'user_dr_tsne.png')
# plt.show()
plt.cla()
plt.clf()
plt.close()
# Compute DBSCAN
#db = DBSCAN(eps=12, min_samples=10).fit(reduced_data) # 低维数据
db = DBSCAN(eps=4, min_samples=10).fit(X) # 高维数据
labels = db.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
print labels
print n_clusters_
print len(labels[labels>=0])

colors = [c[int(i)] for i in labels]
plt.scatter(reduced_data[:, 0], reduced_data[:, 1], 20, colors) # 20为散点的直径
plt.savefig(cur_file_dir+'/result/'+'user_cluster_dbscan.png')
# plt.show()
plt.cla()
plt.clf()
plt.close()

## 展示用户聚类结果（散点图和excel表格）
USERS = np.loadtxt('/Users/dujiawei/git/UserAnalysis/result/user.txt')
USERS_CLS =  np.hstack((USERS.reshape(USERS.size, 1),labels.reshape(labels.size,1)))
print USERS_CLS.shape
np.savetxt('user_cls.txt', USERS_CLS, fmt='%d')
USERS_CLS_DF = pd.DataFrame(USERS_CLS, columns=['USER_ID', 'CLS_ID'])
print USERS_CLS_DF.shape
USERS_CLS_DF.to_sql('qyw_7th_user_clusters', conn, flavor='mysql', if_exists='replace', index=False)
USERS_CLS_CNTCASE = pd.read_sql('''
    SELECT t1.*,
       t2.CLS_ID
FROM
  (SELECT USER_ID,
          COUNT(DISTINCT CASE_ID) AS CNT_CASE
   FROM qyw_7th_yy_succ_all_selected
   GROUP BY USER_ID) AS t1
INNER JOIN
  (SELECT *
   FROM qyw_7th_user_clusters) AS t2 ON t1.USER_ID = t2.USER_ID
ORDER BY t1.CNT_CASE DESC;
''', con=conn)
colors = [c[int(i)] for i in np.array(USERS_CLS_CNTCASE['CLS_ID'])]
plt.scatter(USERS_CLS_CNTCASE['CNT_CASE'], USERS_CLS_CNTCASE['USER_ID'], 20, colors) # 20为散点的直径
plt.savefig(cur_file_dir+'/result/'+'user_cls_cntcase.png')
# plt.show()
plt.cla()
plt.clf()
plt.close()
USERS_CLS_CNTCASE.to_excel(cur_file_dir+'/result/'+'user_cls_cntcase.xls')
conn.close()