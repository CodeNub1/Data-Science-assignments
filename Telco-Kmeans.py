# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 17:51:01 2022

@author: Sidha
"""

import pandas as pd
import matplotlib.pyplot as plt



#reading specific sheet from worksheet
df=pd.read_excel('C:/Users/Sidha/OneDrive/Desktop/Datasets/Clustering/Telco_customer_churn.xlsx')
df.info()



#dropping ID column 
df1=df.drop(['Customer ID'],axis=1)
df1=df1.drop(['Quarter'],axis=1)
df1=df1.drop(['Count'],axis=1)

df1.info()
df1.shape



#separating categorical and numerical
df_cat=df1.iloc[:,[0,3,4,6,7,8,10,11,12,13,14,15,16,17,18,19,20]]
df_num=df1.iloc[:,[1,2,5,9,21,22,23,24,25,26]]




#Dummy creation for categorical
from sklearn.preprocessing import LabelEncoder
col=df_cat.columns.tolist()
i=0
for i in col:
   df_cat = pd.get_dummies(df_cat, columns =[i] ,drop_first = True)    



#concatenating the categorical and numerical datasets
df1=pd.concat([df_cat,df_num],axis=1)
df1.shape

#Normalizing the data using standard scaler
from sklearn.preprocessing import StandardScaler
df1_norm=StandardScaler().fit_transform(df1)
df1_norm



#using elbow plot to find optimim number of clusters-K value
from sklearn.cluster import KMeans
awss=[]
k=range(1,35)

for i in k:
    kmeans=KMeans(n_clusters=i)
    kmeans.fit(df1_norm)
    awss.append(kmeans.inertia_)


awss



#creating elbow curve
k=range(1,35)

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
    kmeans.fit(df1_norm)
    score = silhouette_score(df1_norm, kmeans.labels_)
    k = k
    Sil_coff = score
    silhouette_coefficients.append([k, Sil_coff])

silhouette_coefficients





#building clusters with K=4
clusters=KMeans(4,random_state=30).fit(df1_norm)
clusters

clusters.labels_







#assigning cluster labels to the data set
df['clusters']=clusters.labels_



#group data by clsuters
a=df.groupby('clusters').agg(['mean']).reset_index()

