# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 17:51:01 2022

@author: Sidha
"""

import pandas as pd



#reading specific sheet from worksheet
df=pd.read_csv('C:/Users/Sidha/OneDrive/Desktop/Datasets/Clustering/AutoInsurance.csv')
df.info()
df.columns


#dropping columns 
df1=df.drop(['Customer'],axis=1)
df1=df1.drop(['State'],axis=1)
df1=df1.drop(['Effective To Date'],axis=1)
df1.info()



#separating categorical and numerical data
df_cat=df1.iloc[:,[1,2,3,4,5,7,8,14,15,16,17,19,20]]
df_num=df1.iloc[:,[6,9,10,11,12,13,18]]



#creating dummy variables
from sklearn.preprocessing import LabelEncoder
col=df_cat.columns.tolist()
a=len(col)
i=0
for i in col:
   df_cat = pd.get_dummies(df_cat, columns =[i] ,drop_first = True)    

#concatenating the data frames
y=pd.concat([df_cat,df_num],axis=1)


#Normalizing the data
from sklearn.preprocessing import normalize
df_norm=pd.DataFrame(normalize(y),columns=y.columns)







#Creating Dendrogram based on complete linkage wethod
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt
plt.figure(1, figsize = (16, 8))
dg=dendrogram(linkage(y,method='ward'))



#creating Clusters
from sklearn.cluster import AgglomerativeClustering 
clust=AgglomerativeClustering(n_clusters=2,affinity='euclidean',linkage='complete')

y=clust.fit_predict(df_norm)
clust.labels_

clust_labels=pd.Series(clust.labels_)

#adding cluster to original data set
df['cluster'] = clust_labels

df.head()





#sorting the data set according to clusters
a=df.groupby('cluster').agg(['mean']).reset_index()
