# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 17:51:01 2022

@author: Sidha
"""

import pandas as pd
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.cluster import AgglomerativeClustering 
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from sklearn import preprocessing
import scipy.cluster.hierarchy as sch


#reading specific sheet from worksheet
xls=pd.ExcelFile('C:/Users/Sidha/OneDrive/Desktop/Datasets/Clustering/EastWestAirlines.xlsx')
df=pd.read_excel(xls,'data')
df.info()



#dropping ID column 
df1=df.drop(['ID#'],axis=1)
df1.info()


#Normalizing the data
df1_norm=pd.DataFrame(normalize(df1),columns=df1.columns)


#Creating Dendrogram based on complete linkage wethod
plt.figure(1, figsize = (16, 8))
dg=dendrogram(linkage(df1_norm,method='complete'))



#creating Clusters
clust=AgglomerativeClustering(n_clusters=5,affinity='euclidean',linkage='complete')

y=clust.fit_predict(df1_norm)
clust.labels_

clust_labels=pd.Series(clust.labels_)

#adding cluster to original data set
df['cluster'] = clust_labels

df.head()





#sorting the data set according to clusters
a=df.groupby('cluster').agg(['mean']).reset_index()
