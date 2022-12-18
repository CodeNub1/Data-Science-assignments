# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 17:51:01 2022

@author: Sidha
"""

import pandas as pd
import matplotlib.pyplot as plt



#reading specific sheet from worksheet
xls=pd.ExcelFile('C:/Users/Sidha/OneDrive/Desktop/Datasets/Clustering/EastWestAirlines.xlsx')
df=pd.read_excel(xls,'data')
df.info()



#dropping ID column 
df1=df.drop(['ID#'],axis=1)
df1.info()


#Normalizing the data using standard scaler
from sklearn.preprocessing import StandardScaler
df1_norm=StandardScaler().fit_transform(df1)
df1_norm.dtype





#using elbow plot to find optimim number of clusters-K value
#Creating Dendrogram based on complete linkage wethod
from sklearn.cluster import KMeans
twss=[]
k=range(1,11)

for i in k:
    kmeans=KMeans(n_clusters=i,random_state=2)
    kmeans.fit(df1_norm)
    twss.append(kmeans.inertia_)


twss



#creating elbow curve
plt.plot(k,twss)
plt.title('Elbow Graph')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()



#evaluation clusters using sihouette coefficient
from sklearn.metrics import silhouette_score

silhouette_coefficients = []

for k in range (2, 11):
    kmeans = KMeans(n_clusters = k, init = "random", random_state = 1)
    kmeans.fit(df1_norm)
    score = silhouette_score(df1_norm, kmeans.labels_)
    k = k
    Sil_coff = score
    silhouette_coefficients.append([k, Sil_coff])

silhouette_coefficients
'''
K=2 is the best value for clusters
'''



#building clusters with K=2
clusters=KMeans(2,random_state=30).fit(df1_norm)
clusters

clusters.labels_







#assigning cluster labels to the data set
df['clusters']=clusters.labels_



#group data by clsuters
a=df.groupby('clusters').agg(['mean']).reset_index()

