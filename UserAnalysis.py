#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import offsetbox
from sklearn import (manifold, datasets, decomposition, ensemble,
                     discriminant_analysis, random_projection)
from sklearn.cluster import KMeans, DBSCAN
from scipy.spatial import distance

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

cur_file_dir = cur_file_dir() + '/'
#cur_file_dir = ''

## 读取数据（user_dim.txt）
X = np.loadtxt('/Users/dujiawei/git/UserAnalysis/result/user_dim.txt')

##归一化
#nrow, _ = X.shape
#for row in xrange(nrow):
#    X[row] = X[row]/sum(X[row])
#X = np.vectorize(lambda x: round(x,2))(X)

# Open database connection
conn = MySQLdb.connect(host='localhost',user='root',passwd='',db='qyw', charset='utf8')
#c = ['yellowgreen', 'lightskyblue', 'lightcoral', 'gold', 'blue']

## t-SNE算法降维+DBSCAN聚类（效果最好）
# ###############################################################################
# not good
# Compute Kmeans
# Visualize the results on PCA-reduced data
#tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
#reduced_data = tsne.fit_transform(X)
#plt.scatter(reduced_data[:, 0], reduced_data[:, 1])
#plt.savefig(cur_file_dir+'result/'+'user_dr_tsne.png')
## plt.show()
#plt.cla()
#plt.clf()
#plt.close()
## compute K-means
#kmeans = KMeans(init='k-means++', n_clusters=4, n_init=1)
#cls = kmeans.fit(reduced_data)
#print cls.cluster_centers_
#print cls.labels_
#print cls.inertia_
#labels = cls.labels_

###############################################################################
## Visualize the results on PCA-reduced data
tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
reduced_data = tsne.fit_transform(X)
plt.scatter(reduced_data[:, 0], reduced_data[:, 1])
plt.savefig(cur_file_dir+'result/'+'user_dr_tsne.png')
# plt.show()
plt.cla()
plt.clf()
plt.close()
# Compute DBSCAN
D = distance.squareform(distance.pdist(X)) # 高维数据
#D = distance.squareform(distance.pdist(reduced_data)) # 低维数据
D = np.sort(D,axis=0)
minPts = 20
nearest = D[1:(minPts+1), :]
nearest = nearest.reshape(1, nearest.size)
sort_nearest = np.sort(nearest)
plt.plot(range(len(sort_nearest[0,:])), sort_nearest[0,:], linewidth=1.0, marker='x')
#plt.axis([-2, len(sort_nearest[0,:])+1000, -2, max(sort_nearest[0,:])+2])
plt.savefig(cur_file_dir+'result/'+'nearest.png')
plt.cla()
plt.clf()
plt.close()
#db = DBSCAN(eps=4, min_samples=minPts).fit(X) # 高维数据
db = DBSCAN(eps=5, min_samples=minPts).fit(reduced_data) # 低维数据
labels = db.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
print labels
print n_clusters_
print len(labels[labels>=0])

## 散点图
#colors = [c[int(i) % len(c)] for i in labels]
#colors = labels
#plt.scatter(reduced_data[:, 0], reduced_data[:, 1], 20, colors) # 20为散点的直径

## 图形展示
unique_labels = set(labels)
colors = plt.cm.Spectral(np.linspace(0,1,len(unique_labels)))
for k, col in zip(unique_labels, colors):
    if k  == -1:
        # 噪声显示为黑色
        col = 'k'
        markersize = 3
    else:
        markersize = 8
    class_member_mask = (labels == k)
    xy = reduced_data[class_member_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,   
             markeredgecolor='k', markersize=markersize)   
    plt.title('Estimated number of clusters: %d' % n_clusters_)
plt.savefig(cur_file_dir+'result/'+'user_cluster_dbscan.png')
# plt.show()
plt.cla()
plt.clf()
plt.close()

## 展示用户聚类结果（散点图和excel表格）
USERS = np.loadtxt('/Users/dujiawei/git/UserAnalysis/result/user.txt')
USERS_CLS =  np.hstack((USERS.reshape(USERS.size, 1),labels.reshape(labels.size,1)))
print USERS_CLS.shape
np.savetxt(cur_file_dir+'result/user_cls.txt', USERS_CLS, fmt='%d')
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
#colors = [c[int(i) % len(c)] for i in np.array(USERS_CLS_CNTCASE['CLS_ID'])]
#colors = USERS_CLS_CNTCASE['CLS_ID']
#plt.scatter(USERS_CLS_CNTCASE['CNT_CASE'], USERS_CLS_CNTCASE['USER_ID'], 20, colors) # 20为散点的直径
## 图形展示
labels = np.array(USERS_CLS_CNTCASE['CLS_ID'])
unique_labels = set(labels)
colors = plt.cm.Spectral(np.linspace(0,1,len(unique_labels)))
cntcases = np.array(USERS_CLS_CNTCASE['CNT_CASE'])
cntcases = cntcases.reshape(cntcases.size, 1)
uids = np.array(USERS_CLS_CNTCASE['USER_ID'])
uids = uids.reshape(uids.size, 1)
data = np.hstack((cntcases, uids))
plt.axis([min(cntcases)-2, max(cntcases)+2, min(uids)-1000000, max(uids)+1000000]) #hard code, +- 为x轴、y轴分割（一格）单位
for k, col in zip(unique_labels, colors):
    if k  == -1:
        # 噪声显示为黑色
        col = 'k'
        markersize = 3
    else:
        markersize = 8
    class_member_mask = (labels == k)
    xy = data[class_member_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,   
             markeredgecolor='k', markersize=markersize) 
plt.savefig(cur_file_dir+'result/'+'user_cls_cntcase.png')
# plt.show()
plt.cla()
plt.clf()
plt.close()
USERS_CLS_CNTCASE.to_excel(cur_file_dir+'result/'+'user_cls_cntcase.xls')

## 年龄分布(optional)
#AGE_ALL = pd.read_sql('''
#SELECT AGE FROM qyw_7th_user;
#''',con=conn)
#AGE_ALL = AGE_ALL.AGE
#AGE_ALL = AGE_ALL.fillna(-1)
#AGE_ALL = AGE_ALL.sort_values()
#plt.scatter(range(AGE_ALL.size),AGE_ALL)
#plt.show()

## 特征提取（分类）
USER_INFO_CLS = pd.read_sql('''
SELECT t1.GENDER,
       t1.MEDICAL_GUIDE AS USER_TYPE,
       t1.CITY,
       t1.AGE,
       t2.CLS_ID
FROM
  (SELECT *
   FROM qyw_7th_user
   ORDER BY USER_ID) AS t1
INNER JOIN
  (SELECT *
   FROM qyw_7th_user_clusters
   WHERE CLS_ID >= 0
   ORDER BY USER_ID) AS t2 ON t1.USER_ID = t2.USER_ID;''', con=conn)
USER_INFO_CLS.loc[USER_INFO_CLS['GENDER'] == 0, 'GENDER'] = 'UNKNOWN'
USER_INFO_CLS.loc[USER_INFO_CLS['GENDER'] == 1, 'GENDER'] = 'MALE'
USER_INFO_CLS.loc[USER_INFO_CLS['GENDER'] == 2, 'GENDER'] = 'FEMALE'
USER_INFO_CLS.loc[USER_INFO_CLS['USER_TYPE'] != '', 'USER_TYPE'] = 'MEDICAL_GUIDED'
USER_INFO_CLS.loc[USER_INFO_CLS['USER_TYPE'] == '', 'USER_TYPE'] = 'SELF_GUIDED'
USER_INFO_CLS['AGE'].fillna(-1, inplace=True)
USER_INFO_CLS['CITY'].fillna(u'未知', inplace=True)
USER_INFO_CLS.loc[USER_INFO_CLS['CITY'] != u'武汉', 'CITY'] = 'HOSPITAL LOCATED'
USER_INFO_CLS.loc[USER_INFO_CLS['CITY'] == u'武汉', 'CITY'] = 'OTHERS'

# 年龄离散化（optional）
USER_INFO_CLS.AGE = USER_INFO_CLS.AGE.apply(lambda x:'{begin} TO {end}'.format(begin=int(x)/10*10,end=(int(x)/10+1)*10))
USER_INFO_CLS.loc[USER_INFO_CLS['AGE'] == '-10 -> 0', 'AGE'] = 'UNKNOWN'

# encoding
USER_INFO_CLS_FEA = USER_INFO_CLS.drop(['CLS_ID'], axis=1)
USER_INFO_CLS_FEA_DICT = USER_INFO_CLS_FEA.T.to_dict().values()

from sklearn.feature_extraction import DictVectorizer
vec = DictVectorizer()
USER_INFO_CLS_DATA = vec.fit_transform(USER_INFO_CLS_FEA_DICT).toarray()
USER_INFO_CLS_FEA_NAMES = vec.get_feature_names()
print USER_INFO_CLS_FEA_NAMES
USER_INFO_CLS_TARGET = USER_INFO_CLS.CLS_ID.values

from sklearn import tree
clf = tree.DecisionTreeClassifier()
clf = clf.fit(USER_INFO_CLS_DATA, USER_INFO_CLS_TARGET)

from sklearn.externals.six import StringIO 
import pydot 

# dot_data = StringIO() 
# tree.export_graphviz(clf, out_file=dot_data) 
# graph = pydot.graph_from_dot_data(dot_data.getvalue()) 
# graph.write_png(cur_file_dir+"result/USER_INFO_CLS_clf_1.png") 

USER_INFO_CLS_TARGET = ['cls #'+str(int(i)) for i in np.unique(USER_INFO_CLS_TARGET)]
print USER_INFO_CLS_TARGET
from IPython.display import Image  
dot_data = StringIO()  
tree.export_graphviz(clf, out_file=dot_data,  
                         feature_names=USER_INFO_CLS_FEA_NAMES,  
                         class_names=USER_INFO_CLS_TARGET,  
                         filled=True, rounded=True,  
                         special_characters=False)  # (default=False) When set to False, ignore special characters for PostScript compatibility.
graph = pydot.graph_from_dot_data(dot_data.getvalue())  
graph.write_png(cur_file_dir+"result/user_info_cls_clf.png")
# conn.close()

## 特征提取（预测）
USER_INFO_CLS = pd.read_sql('''
SELECT t1.GENDER,
       t1.MEDICAL_GUIDE AS USER_TYPE,
       t1.CITY,
       t1.AGE,
       t2.CLS_ID
FROM
  (SELECT *
   FROM qyw_7th_user
   ORDER BY USER_ID) AS t1
INNER JOIN
  (SELECT *
   FROM qyw_7th_user_clusters
   WHERE CLS_ID >= 0
   ORDER BY USER_ID) AS t2 ON t1.USER_ID = t2.USER_ID;''', con=conn)
USER_INFO_CLS.loc[USER_INFO_CLS['GENDER'] == 0, 'GENDER'] = 'UNKNOWN'
USER_INFO_CLS.loc[USER_INFO_CLS['GENDER'] == 1, 'GENDER'] = 'MALE'
USER_INFO_CLS.loc[USER_INFO_CLS['GENDER'] == 2, 'GENDER'] = 'FEMALE'
USER_INFO_CLS.loc[USER_INFO_CLS['USER_TYPE'] != '', 'USER_TYPE'] = 'MEDICAL_GUIDED'
USER_INFO_CLS.loc[USER_INFO_CLS['USER_TYPE'] == '', 'USER_TYPE'] = 'SELF_GUIDED'
USER_INFO_CLS['AGE'].fillna(-1, inplace=True)
USER_INFO_CLS['CITY'].fillna(u'未知', inplace=True)
USER_INFO_CLS.loc[USER_INFO_CLS['CITY'] != u'武汉', 'CITY'] = 'HOSPITAL LOCATED'
USER_INFO_CLS.loc[USER_INFO_CLS['CITY'] == u'武汉', 'CITY'] = 'OTHERS'

# # 年龄离散化（optional）
# USER_INFO_CLS.AGE = USER_INFO_CLS.AGE.apply(lambda x:'{begin} TO {end}'.format(begin=int(x)/10*10,end=(int(x)/10+1)*10))
# USER_INFO_CLS.loc[USER_INFO_CLS['AGE'] == '-10 -> 0', 'AGE'] = 'UNKNOWN'

# encoding
USER_INFO_CLS_FEA = USER_INFO_CLS.drop(['CLS_ID'], axis=1)
USER_INFO_CLS_FEA_DICT = USER_INFO_CLS_FEA.T.to_dict().values()

from sklearn.feature_extraction import DictVectorizer
vec = DictVectorizer()
USER_INFO_CLS_DATA = vec.fit_transform(USER_INFO_CLS_FEA_DICT).toarray()
USER_INFO_CLS_FEA_NAMES = vec.get_feature_names()
# print USER_INFO_CLS_FEA_NAMES
USER_INFO_CLS_TARGET = USER_INFO_CLS.CLS_ID.values

# 将数据集分为训练集和测试集
from sklearn.cross_validation import train_test_split
data_train, data_test, target_train, target_test = train_test_split(USER_INFO_CLS_DATA, USER_INFO_CLS_TARGET)
print len(data_train), len(data_test)

# 这里选择朴素贝叶斯、决策树、随机森林和SVM来做一个对比。
from sklearn import cross_validation
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
import datetime
estimators = {}
estimators['bayes'] = GaussianNB()
estimators['tree'] = tree.DecisionTreeClassifier()
estimators['forest_100'] = RandomForestClassifier(n_estimators = 100)
estimators['forest_10'] = RandomForestClassifier(n_estimators = 10)
estimators['svm_c_rbf'] = svm.SVC()
estimators['svm_c_linear'] = svm.SVC(kernel='linear')
estimators['svm_linear'] = svm.LinearSVC()
# estimators['svm_nusvc'] = svm.NuSVC() # error, because: http://stackoverflow.com/questions/26987248/nu-is-infeasible

for k in estimators.keys():
    start_time = datetime.datetime.now()
    print '----%s----' % k
    estimators[k] = estimators[k].fit(data_train, target_train)
    pred = estimators[k].predict(data_test)
    # print target_test vs. pred
#     print target_test
#     print 'vs. '
#     print pred
    # This is predict score
    print("%s Score: %0.2f" % (k, estimators[k].score(data_test, target_test)))
    scores = cross_validation.cross_val_score(estimators[k], data_test, target_test, cv=5)
    print("%s Cross Avg. Score: %0.2f (+/- %0.2f)" % (k, scores.mean(), scores.std() * 2))
#     print("%s Pred Score: %0.2f" % (k, ((float)(len(target_test[target_test == pred])))/len(target_test)))
    end_time = datetime.datetime.now()
    time_spend = end_time - start_time
    print("%s Time: %0.2f" % (k, time_spend.total_seconds()))
    
conn.close()