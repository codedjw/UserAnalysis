#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import scipy.cluster.hierarchy as hcluster
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt

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

## 绘制dendrogram （根据linkage）
def fancy_dendrogram(*args, **kwargs):
    max_d = kwargs.pop('max_d', None)
    if max_d and 'color_threshold' not in kwargs:
        kwargs['color_threshold'] = max_d
    annotate_above = kwargs.pop('annotate_above', 0)

    ddata = dendrogram(*args, **kwargs)

    if not kwargs.get('no_plot', False):
        plt.title('Hierarchical Clustering Dendrogram (truncated)')
        plt.xlabel('sample index or (cluster size)')
        plt.ylabel('distance')
        for i, d, c in zip(ddata['icoord'], ddata['dcoord'], ddata['color_list']):
            x = 0.5 * sum(i[1:3])
            y = d[1]
            if y > annotate_above:
                plt.plot(x, y, 'o', c=c)
                plt.annotate("%.3g" % y, (x, y), xytext=(0, -5),
                             textcoords='offset points',
                             va='top', ha='center')
        if max_d:
            plt.axhline(y=max_d, c='k')
    return ddata

## Case Hierarchy Cluster (Step01 - Ouput: dendrogram)
# 读入Case Similarity Matrix(小数点后两位)
case_sim_matrix = np.loadtxt("/Users/dujiawei/git/UserAnalysis/result/case_sim.txt")

case_dist_matrix = 1 - case_sim_matrix
print case_sim_matrix.mean(), case_sim_matrix.max(), case_sim_matrix.min()
print case_dist_matrix.mean(), case_dist_matrix.max(), case_dist_matrix.min()
pairwise_dist = []
print case_dist_matrix
nrow, ncol = case_dist_matrix.shape
for i in xrange(nrow):
    for j in xrange(i+1,ncol):
        pairwise_dist.append(round(case_dist_matrix[i][j],2))
linkage_matrix = linkage(pairwise_dist, method='average')
plt.figure(figsize=(20,10))
#max_d = 1.19 # 120 gap = -1
#max_d = 1.38 # 1000 gap = -1
# max_d = 0.5 # 1000 gap = 0
max_d = 0.5 # 120 gap = 0
fancy_dendrogram(
    linkage_matrix,
    truncate_mode='lastp',
    p=30,
    leaf_rotation=90.,
    leaf_font_size=12.,
    show_contracted=True,
    annotate_above=0.1,
    max_d=max_d,
)
plt.savefig(cur_file_dir+'result/'+'case_dendrogram.png')
#plt.show()
plt.cla()
plt.clf()
plt.close()

# last = linkage_matrix[-10:, 2]
# last_rev = last[::-1]
# idxs = np.arange(1, len(last) + 1)
# plt.plot(idxs, last_rev)

# acceleration = np.diff(last, 2)  # 2nd derivative of the distances
# acceleration_rev = acceleration[::-1]
# plt.plot(idxs[:-2] + 1, acceleration_rev)
# plt.show()
# k = acceleration_rev.argmax() + 2  # if idx 0 is the max of this we want 2 clusters
# print "clusters:", k

## Case Hierarchy Cluster (Step02 - Ouput: Cluster)
# max_d = 0.4775 
# cr = 'distance'
# # k = 60
# # cr = 'maxclust'
cr = 'distance'
clusters = hcluster.fcluster(linkage_matrix, max_d, criterion=cr) # fcluster取得 <= max_d (dendrogram不太准确，取得是<max_d)
# print np.sort(clusters)
np.savetxt(cur_file_dir+'result/'+'case_cluster.txt',clusters.reshape(nrow,1), fmt='%d')