# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 17:51:01 2022

@author: Sidha
"""

import pandas as pd
import numpy as np


#imprting worksheet
df=pd.read_csv('C:/Users/Sidha/OneDrive/Desktop/Datasets/Clustering/AutoInsurance.csv')
df.info()
df.shape

#dropping columns 
df1=df.drop(['Customer'],axis=1)
df1=df1.drop(['State'],axis=1)
df1=df1.drop(['Effective To Date'],axis=1)
df1.info()
df1.shape
#separating numerical and categorical data for encoding
df1_cat=df1.iloc[:,[1,2,3,4,5,7,8,14,15,16,17,19,20]]
df1_num=df1.iloc[:,[6,9,10,11,12,13,18]]



#creating labels for the categorical features
from sklearn.preprocessing import LabelEncoder
col=df1_cat.columns.tolist()

i=0
for i in col:
   df1_cat = pd.get_dummies(df1_cat, columns =[i] ,drop_first = True)    




#concatenating  num and categorical DataFrames
y=pd.concat([df1_num,df1_cat],axis=1)
y.shape



#Normalizing the data
from sklearn.preprocessing import normalize
y_norm=pd.DataFrame(normalize(y),columns=y.columns)
y_norm.shape



#using elbow plot to find optimim number of clusters-K value
from sklearn.cluster import KMeans
awss=[]
k=range(1,46)

for i in k:
    kmeans=KMeans(n_clusters=i)
    kmeans.fit(y_norm)
    awss.append(kmeans.inertia_)


awss



#creating elbow curve
import matplotlib.pyplot as plt
k=range(1,45)

plt.plot(k,awss)
plt.title('Elbow Graph')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()



#evaluation the number of clusters
from sklearn.metrics import silhouette_score

silhouette_coefficients = []

for k in range (2, 10):
    kmeans = KMeans(n_clusters = k, init = "random", random_state = 1)
    kmeans.fit(y_norm)
    score = silhouette_score(y_norm, kmeans.labels_)
    k = k
    Sil_coff = score
    silhouette_coefficients.append([k, Sil_coff])

silhouette_coefficients



#building clusters with K=2
clusters=KMeans(2,random_state=30).fit(y_norm)
clusters

clusters.labels_

#adding cluster to original data set
df['cluster'] = clusters.labels_


df.shape




#sorting the data set according to clusters
a = df.groupby('cluster').agg(['mean']).reset_index()
a.shape
