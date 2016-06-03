
# coding: utf-8

# In[6]:

import numpy as np
import scipy.cluster.hierarchy as hcluster
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt


# ## 获取当前路径

# In[7]:

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


# ## 绘制dendrogram （根据linkage）

# In[8]:

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


# ## Case Hierarchy Cluster (Step01 - Ouput: dendrogram)

# In[9]:

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


# ## Case Hierarchy Cluster (Step02 - Ouput: Cluster)

# In[10]:

# max_d = 0.4775 
# cr = 'distance'
# # k = 60
# # cr = 'maxclust'
cr = 'distance'
clusters = hcluster.fcluster(linkage_matrix, max_d, criterion=cr) # fcluster取得 <= max_d (dendrogram不太准确，取得是<max_d)
# print np.sort(clusters)
np.savetxt(cur_file_dir+'result/'+'case_cluster.txt',clusters.reshape(nrow,1), fmt='%d')


# # # Example-01: scipy.cluster.hierarchy.linkage & scipy.spatial.distance.pdist

# # In[ ]:

# #open the file assuming the data above is in a file called 'dataFile'
# inFile = open('dataFile','r')
# #save the column/row headers (conditions/genes) into an array
# colHeaders = inFile.next().strip().split()[1:]
# rowHeaders = []
# dataMatrix = []

# for line in inFile:
#     data = line.strip().split(' ')
#     rowHeaders.append(data[0])
#     dataMatrix.append([float(x) for x in data[1:]])

# #convert native data array into a numpy array
# dataMatrix = np.array(dataMatrix) 
# distanceMatrix = pdist(dataMatrix)
# # distanceMatrix = pdist(dataMatrix,'hamming') #use hamming function
# # distanceMatrix = pdist(dataMatrix,'euclidean') #use euclidean function
# linkageMatrix = linkage(distanceMatrix, method='single')
# dataMatrix
# distanceMatrix
# linkageMatrix


# # ## Example-02: scipy.cluster.hierarchy.linkage & dendrogram（实例）

# # In[ ]:

# # generate two clusters: a with 100 points, b with 50:
# np.random.seed(4711)  # for repeatability of this tutorial
# a = np.random.multivariate_normal([10, 0], [[3, 1], [1, 4]], size=[100,])
# b = np.random.multivariate_normal([0, 20], [[3, 1], [1, 4]], size=[50,])
# X = np.concatenate((a, b),)
# print X.shape  # 150 samples with 2 dimensions
# plt.scatter(X[:,0], X[:,1])
# plt.show()
# # generate the linkage matrix
# Z = linkage(X, 'ward')
# fancy_dendrogram(
#     Z,
#     truncate_mode='lastp',
#     p=12,
#     leaf_rotation=90.,
#     leaf_font_size=12.,
#     show_contracted=True,
#     annotate_above=10,  # useful in small plots so annotations don't overlap
# )
# plt.show()


# # In[ ]:



