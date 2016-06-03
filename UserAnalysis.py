
# coding: utf-8

# In[177]:

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
import random


# ## 获取当前路径

# In[6]:

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
# cur_file_dir = ''


# ## 读取数据（user_dim.txt）

# In[175]:

## 读取数据（user_dim.txt）
X = np.loadtxt('/Users/dujiawei/git/UserAnalysis/result/user_dim.txt')

##归一化
# nrow, _ = X.shape
# for row in xrange(nrow):
#     X[row] = X[row]/sum(X[row])
# X = np.vectorize(lambda x: round(x,2))(X)

# Open database connection
conn = MySQLdb.connect(host='localhost',user='root',passwd='',db='qyw', charset='utf8')
#c = ['yellowgreen', 'lightskyblue', 'lightcoral', 'gold', 'blue']


# ## t-SNE算法降维+DBSCAN聚类（效果最好）

# In[189]:

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
#D = distance.squareform(distance.pdist(X)) # 高维数据
D = distance.squareform(distance.pdist(reduced_data)) # 低维数据
D = np.sort(D,axis=0)
minPts = 10
nearest = D[1:(minPts+1), :]
nearest = nearest.reshape(1, nearest.size)
sort_nearest = np.sort(nearest)
plt.plot(range(len(sort_nearest[0,:])), sort_nearest[0,:], linewidth=1.0, marker='x')
#plt.axis([-2, len(sort_nearest[0,:])+1000, -2, max(sort_nearest[0,:])+2])
plt.savefig(cur_file_dir+'result/'+'nearest.png')
plt.cla()
plt.clf()
plt.close()
#db = DBSCAN(eps=0.90, min_samples=minPts).fit(X) # 高维数据
db = DBSCAN(eps=30, min_samples=minPts).fit(reduced_data) # 低维数据
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
f = open(cur_file_dir+'result/cls_special_user_dim.txt', 'w')
for k, col in zip(unique_labels, colors):
    if k  == -1:
        # 噪声显示为黑色
        col = 'k'
        markersize = 3
    else:
        markersize = 8
    class_member_mask = (labels == k)
    xy = reduced_data[class_member_mask]
    if k >= 0: # 非离群点
    	xxy = X[class_member_mask]
    	nrow,ncol = xxy.shape
    	np.savetxt(f, xxy[random.randint(0, nrow-1),:].reshape(1,ncol),fmt='%d')
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,   
             markeredgecolor='k', markersize=markersize)   
    plt.title('Estimated number of clusters: %d' % n_clusters_)
f.close()
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


# # ## 年龄分布（年龄映射为标称属性）optional

# # In[17]:

# AGE_ALL = pd.read_sql('''
# SELECT AGE FROM qyw_7th_user;
# ''',con=conn)
# AGE_ALL = AGE_ALL.AGE
# AGE_ALL = AGE_ALL.fillna(-1)
# AGE_ALL = AGE_ALL.sort_values()
# plt.scatter(range(AGE_ALL.size),AGE_ALL)
# plt.show()


# ## 打印分类树规则

# In[163]:

def get_lineage(tree, feature_names, class_names, numerial_feature_names=None):
    left= tree.tree_.children_left
    right= tree.tree_.children_right
    threshold = tree.tree_.threshold
    features  = [feature_names[i] for i in tree.tree_.feature]
    values = tree.tree_.value

    # get ids of child nodes
    idx = np.argwhere(left == -1)[:,0]     

    def recurse(left, right, child, lineage=None):          
        if lineage is None:
            lineage = [child]
        if child in left:
            parent = np.where(left == child)[0].item()
            split = 'l'
        else:
            parent = np.where(right == child)[0].item()
            split = 'r'

        lineage.append((parent, split, threshold[parent], features[parent]))

        if parent == 0:
            lineage.reverse()
            return lineage
        else:
            return recurse(left, right, parent, lineage)
    f = open(cur_file_dir+'result/user_info_cls_rules.txt','w')
    i = 0
    for child in idx:
        str = ''
        for node in recurse(left, right, child):
            if isinstance(node, tuple):
                _,cur_split,cur_threshold,cur_feature = node
                feature_label = cur_feature
                if numerial_feature_names:
                    if cur_feature in numerial_feature_names:
                        comp_label = '<='
                        if cur_split == 'r':
                            comp_label = '>'
                        feature_label = '{lnum}{comp}{thr}'.format(lnum=feature_label, comp=comp_label, thr=cur_threshold)
                and_label = ''
                if str != '':
                    and_label = ' AND '
                not_label = ''
                if cur_split == 'l':
                    not_label = 'NOT '
                    if numerial_feature_names:
                        if cur_feature in numerial_feature_names:
                            not_label = ''
                str = '{origin}{land}({lnot}{lnew})'.format(origin=str, land=and_label, lnot=not_label, lnew=feature_label)
            else:
                value = values[node]
                max_idx = 0
                max_n_samples = 0
                _, v_ncol = value.shape
                for v in xrange(v_ncol):
                    if max_n_samples < value[0,v]:
                        max_n_samples = value[0,v]
                        max_idx = v
                cur_class = class_names[max_idx]
                str = '{lclass}\t{features}'.format(lclass=cur_class, features=str)
#                 print str
                f.write(str)
                if i<len(idx)-1:
                    f.write('\n')
                i = i+1
                str = ''
    f.close()


# ## 特征提取（分类）

# In[164]:

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
USER_INFO_CLS.loc[USER_INFO_CLS['AGE'] == '-10 TO 0', 'AGE'] = 'UNKNOWN'

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

USER_INFO_CLS_TARGET = ['cls_#'+str(int(i)) for i in np.unique(USER_INFO_CLS_TARGET)]
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

get_lineage(clf, USER_INFO_CLS_FEA_NAMES, USER_INFO_CLS_TARGET) # 年龄离散化（optional）
# get_lineage(clf, USER_INFO_CLS_FEA_NAMES, USER_INFO_CLS_TARGET, ['AGE'])

# conn.close()


# ## 特征提取（预测）

# In[42]:

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


# # ## C4.5样例代码

# # In[ ]:

# from sklearn.datasets import load_iris
# from sklearn import tree
# iris = load_iris()
# clf = tree.DecisionTreeClassifier()
# print iris.feature_names
# # clf = clf.fit(iris.data, iris.target)
# # # from sklearn.externals.six import StringIO  
# # # import pydot 
# # # dot_data = StringIO() 
# # # tree.export_graphviz(clf, out_file=dot_data) 
# # # graph = pydot.graph_from_dot_data(dot_data.getvalue()) 
# # # graph.write_pdf("iris.pdf") 
# # from IPython.display import Image  
# # dot_data = StringIO()  
# # tree.export_graphviz(clf, out_file=dot_data,  
# #                          feature_names=iris.feature_names,  
# #                          class_names=iris.target_names,  
# #                          filled=True, rounded=True,  
# #                          special_characters=True)  
# # graph = pydot.graph_from_dot_data(dot_data.getvalue())  
# # graph.write_png("iris.png")


# # ## 其他降维算法

# # In[ ]:

# #----------------------------------------------------------------------
# # Random 2D projection using a random unitary matrix
# print("Computing random projection")
# rp = random_projection.SparseRandomProjection(n_components=2, random_state=42)
# X_projected = rp.fit_transform(X)
# plt.scatter(X_projected[:, 0], X_projected[:, 1])
# plt.show()


# # In[ ]:

# #----------------------------------------------------------------------
# # Projection on to the first 2 principal components
# print("Computing PCA projection")
# X_pca = decomposition.TruncatedSVD(n_components=2).fit_transform(X)
# plt.scatter(X_pca[:, 0], X_pca[:, 1])
# plt.show()


# # In[ ]:

# #----------------------------------------------------------------------
# # Isomap projection of the digits dataset
# print("Computing Isomap embedding")
# n_neighbors = 3
# X_iso = manifold.Isomap(n_neighbors, n_components=2).fit_transform(X)
# print("Done.")
# plt.scatter(X_iso[:, 0], X_iso[:, 1])
# plt.show()


# # In[ ]:

# #----------------------------------------------------------------------
# # Locally linear embedding of the digits dataset
# print("Computing LLE embedding")
# n_neighbors = 3
# clf = manifold.LocallyLinearEmbedding(n_neighbors, n_components=2,
#                                       method='standard')
# X_lle = clf.fit_transform(X)
# print("Done. Reconstruction error: %g" % clf.reconstruction_error_)
# plt.scatter(X_lle[:, 0], X_lle[:, 1])
# plt.show()


# # In[ ]:

# ## good
# #----------------------------------------------------------------------
# # Modified Locally linear embedding of the digits dataset
# print("Computing modified LLE embedding")
# n_neighbors = 3
# clf = manifold.LocallyLinearEmbedding(n_neighbors, n_components=2,
#                                       method='modified')
# X_mlle = clf.fit_transform(X)
# print("Done. Reconstruction error: %g" % clf.reconstruction_error_)
# plt.scatter(X_mlle[:, 0], X_mlle[:, 1])
# plt.show()


# # In[ ]:

# ## good
# #----------------------------------------------------------------------
# # HLLE embedding of the digits dataset
# print("Computing Hessian LLE embedding")
# n_neighbors = 6
# clf = manifold.LocallyLinearEmbedding(n_neighbors, n_components=2,
#                                       method='hessian')
# X_hlle = clf.fit_transform(X)
# print("Done. Reconstruction error: %g" % clf.reconstruction_error_)
# plt.scatter(X_hlle[:, 0], X_hlle[:, 1])
# plt.show()


# # In[ ]:

# ## good
# #----------------------------------------------------------------------
# # LTSA embedding of the digits dataset
# print("Computing LTSA embedding")
# n_neighbors = 100
# clf = manifold.LocallyLinearEmbedding(n_neighbors, n_components=2,
#                                       method='ltsa')
# X_ltsa = clf.fit_transform(X)
# print("Done. Reconstruction error: %g" % clf.reconstruction_error_)
# plt.scatter(X_ltsa[:, 0], X_ltsa[:, 1])
# plt.show()


# # In[ ]:

# ## good
# #----------------------------------------------------------------------
# # MDS  embedding of the digits dataset
# print("Computing MDS embedding")
# clf = manifold.MDS(n_components=2, n_init=1, max_iter=100)
# X_mds = clf.fit_transform(X)
# print("Done. Stress: %f" % clf.stress_)
# plt.scatter(X_mds[:, 0], X_mds[:, 1])
# plt.show()


# # In[ ]:

# ## good
# #----------------------------------------------------------------------
# # Random Trees embedding of the digits dataset
# print("Computing Totally Random Trees embedding")
# hasher = ensemble.RandomTreesEmbedding(n_estimators=200, random_state=0,
#                                        max_depth=5)
# X_transformed = hasher.fit_transform(X)
# pca = decomposition.TruncatedSVD(n_components=2)
# X_reduced = pca.fit_transform(X_transformed)
# plt.scatter(X_reduced[:, 0], X_reduced[:, 1])
# plt.show()


# # In[ ]:

# #----------------------------------------------------------------------
# # Spectral embedding of the digits dataset
# print("Computing Spectral embedding")
# embedder = manifold.SpectralEmbedding(n_components=2, random_state=0,
#                                       eigen_solver="arpack")
# X_se = embedder.fit_transform(X)
# plt.scatter(X_se[:, 0], X_se[:, 1])
# plt.show()


# # In[ ]:

# ## best
# #----------------------------------------------------------------------
# # t-SNE embedding of the digits dataset
# print("Computing t-SNE embedding")
# tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
# X_tsne = tsne.fit_transform(X)
# plt.scatter(X_tsne[:, 0], X_tsne[:, 1])
# plt.show()

