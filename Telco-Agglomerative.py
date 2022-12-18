# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 17:51:01 2022

@author: Sidha
"""

import pandas as pd
import numpy as np


#imprting worksheet
df=pd.read_excel('C:/Users/Sidha/OneDrive/Desktop/Datasets/Clustering/Telco_customer_churn.xlsx')
df.info()
df.shape



#separating numerical and categorical data for encoding
df_cat=df.iloc[:,[3,6,7,9,10,11,13,14,15,16,17,18,19,20,21,22,23]]
df_num=df.iloc[:,[0,1,2,4,5,8,12,24,25,26,27,28,29]]



#creating labels for the categorical features
from sklearn.preprocessing import LabelEncoder
col=df_cat.columns.tolist()
a=len(col)
i=0
for i in col:
   df_cat = pd.get_dummies(df_cat, columns =[i] ,drop_first = True)    



#dropping cutomer id and Quarter and count
df_num.drop('Customer ID',axis=1,inplace= True)
df_num.drop('Count',axis=1,inplace= True)
df_num.drop('Quarter',axis=1,inplace= True)



#concatenating  num and categorical DataFrames
y=pd.concat([df_num,df_cat],axis=1)
y.shape



#Normalizing the data
from sklearn.preprocessing import normalize
y_norm=pd.DataFrame(normalize(y),columns=y.columns)
y_norm.shape

#Creating Dendrogram based on complete linkage wethod
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
plt.figure(1, figsize = (16, 8))
dg=dendrogram(linkage(y_norm,method='ward'))



#creating Clusters
from sklearn.cluster import AgglomerativeClustering 
clust=AgglomerativeClustering(n_clusters=3,affinity='euclidean',linkage='complete')

a=clust.fit_predict(y_norm)
clust.labels_

clust_labels=pd.Series(clust.labels_)

#adding cluster to original data set
df['cluster'] = clust_labels


df.shape




#sorting the data set according to clusters
a = df.groupby('cluster').agg(['mean']).reset_index()
a.shape
