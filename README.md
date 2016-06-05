# UserAnalysis
User Analysis based on operation logs.

执行方法
----
```
>> run Neo->CaseCluster.java
>> python CaseCluster.py
>> run Neo->UserCluster.java
>> python UserAnalysis.py
```
中间结果文件说明（result文件夹下）
----

文件名| 生成文件 | 描述
-----|--------|----
CaseCluster.java|<font color=green><b>case.txt|Unique Case List<br>大小：size(Unique Case)<br>内容：（CaseID（从0开始）\tList<Event.activity>（以空格分隔））
CaseCluster.java|<b>case_sim.txt|Unique Case Similarity Matrix based on [Needleman Algorithm](https://en.wikipedia.org/wiki/Needleman%E2%80%93Wunsch_algorithm) <br>大小：size(Unique Case) * size(Unique Case)）
CaseCluster.py|<font color=red><b>case_dendrogram.png|Unique Case Cluster based on Scipy Hierarchy Cluster Algorithm<br>显示：系统树图（max_d）
CaseCluster.py|<font color=green><b>case_cluster.txt|Unique Case Cluster ID<br>大小：size(Unique Case)<br>顺序：与case.txt和case_sim.txt一致
UserCluster.java|<font color=green><b>user_dim.txt|基于case.txt和case_cluster.txt，对每个用户的所有Case根据聚类结果进行一个频次打分，形成一个打分矩阵<br>大小：size(Unique User ID) * size(Unique Case Cluster ID)）
UserCluster.java|<font color=blue><b>user.txt|Unique User ID<br>大小：size(Unique User ID)<br>顺序：与user_dim.txt一致
UserAnalysis.py|<b>user_dr_tsne.png|sklearn tSNE算法对user_dim.txt降维后结果<br>显示：二维散点图（基于降维后数据）
UserAnalysis.py|<font color=blue><b>nearest.png|<b>对高维或低维数据通过k-nearest距离计算，绘制的k领域距离变化图，用于确定DBSCAN中的eps
UserAnalysis.py|<font color=red><b>cls_special_user_dim.txt|DBSCAN对高维或低维数据形成的簇，每个簇随机选择一个样本（User），用该User的Case簇打分值表示整个簇的行为特性（打分值由user_dim.txt获得）<br><b>可参考<font color=green>case_cluster.txt</font>(CaseID(offset-1), CCID)、<font color=green>case.txt</font>(CaseID, caseContent)以及<font color=green>user_dim.txt</font>(score(CCID))join得到）
UserAnalysis.py|<font color=red><b>user_cluster_dbscan.png|对高维或低维数据通过DBSCAN聚类后的结果<br>显示：二维散点图（基于降维后数据）<br><b>颜色：DBSCAN聚类后的簇标号（可以观察tSNE算法对全局特性保持的效果）
UserAnalysis.py|<font color=blue><b>user_cls.txt|对高维或低维数据通过DBSCAN聚类后的结果，每个用户对应的簇标号<br>大小：size(Unique User) * 2
UserAnalysis.py|<b>user_cls_cntcase.png|USER_ID、UCID、count(Case)间的关系<br>显示：散点图
UserAnalysis.py|<b>user_cls_cntcase.xls|USER_ID、UCID、count(Case)间的关系<br>显示：Excel表格
UserAnalysis.py|<font color=red><b>user_info_cls_rules.txt|<b>将USER的属性和UCID结果，通过sklearn DescionTreeClassifier分类后生成的规则（分类树叶子节点产生的条件）
UserAnalysis.py|<font color=red><b>user_info_cls_clf.png|<b>将USER的属性和UCID结果，通过sklearn DescionTreeClassifier分类后生成的决策树<br>显示：决策树

实验数据
---
<h3>数据选取</h3>
* 120个用户操作记录：选取根据一定时间区间内一个用户Case执行的数量进行排序，选择频次最多的前20个和频次最少的后100个用户操作记录
* 1000个用户操作记录：选取根据一定时间区间内一个用户Case执行的数量进行排序，选择频次最多的前1000个用户的操作记录

<h3>实验参数</h3>
* 详见Params.xlsx

<h3>测试结果</h3>
* 120数据集：result-120-lowdim-without_normalization/
* 1000数据集：result-1000-lowdim-without_normalization/

Change Log
----------
#### v1.0.0 (2016/06/03 17:00 +08:00)
* [基于业务场景的Case分隔](https://github.com/codedjw/XESConverter) （[基于Case时间间隔/5 min的初次划分](https://github.com/codedjw/DataAnalysis/blob/master/QYW_7th_Analysis/qyw_7th_XESConverterHelp.ipynb)）
* 通过[Needleman算法](https://en.wikipedia.org/wiki/Needleman%E2%80%93Wunsch_algorithm)计算case间相似度
* 基于case间相似度，通过层次聚类获得case的簇（CCID）
* 根据case的簇，对用户的操作记录（Case）进行频次打分，提取用户操作信息（基于case簇，降维）
* 基于用户操作记录的频次打分表，通过tSNE进行降维，获得可视化结果
* 基于用户操作记录的频次打分表（tSNE降维结果），通过DBSCAN对高（低）维数据进行聚类，形成用户簇（UCID）（该用户簇的行为特点由随机选取的样本User的频率打分表表示）
* 基于形成的用户簇（UCID）和用户属性（预处理、离散化），通过决策树算法CART分类（分类器），获得生成的决策树和规则
* 基于形成的用户簇（UCID）和用户属性（预处理、离散化），通过十字交叉验证划分训练集和测试集，评估不同分类算法的效果（Predict Score, Cross Avg Score, Time...）
* 以上两步形成的分类器可以用于基于用户属性的行为预测，即通过分类器获得UCID标号，找出该UCID标号表示的用户簇的行为特点（该用户簇的行为特点由随机选取的样本User的频率打分表表示）