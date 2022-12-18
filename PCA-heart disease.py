# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

'''
USING HEIRARCHICAL CLUSTERING
'''
#importing dataset
df=pd.read_csv('C:/Users/Sidha/OneDrive/Desktop/Datasets/PCA/heart disease.csv')
df.describe()
df.info()

df.isna().sum()


#Normalizing the data
from sklearn.preprocessing import normalize
df_norm=pd.DataFrame(normalize(df),columns=df.columns)



#Creating Dendrogram
from scipy.cluster.hierarchy import linkage, dendrogram
plt.figure(1, figsize = (16, 8))
dg=dendrogram(linkage(df_norm,method='complete'))


#creating clusters
from sklearn.cluster import AgglomerativeClustering 
clust1=AgglomerativeClustering(n_clusters=3,affinity='euclidean',linkage='ward')

y=clust1.fit_predict(df_norm)
clust1.labels_


#Clusters=3 or 2


'''
USING KMEANS
'''
from sklearn.cluster import KMeans

#Kmeans
wcss=[]
for i in range (1,6):
    kmeans=KMeans(n_clusters=i,random_state=2)
    kmeans.fit(df_norm)
    wcss.append(kmeans.inertia_)



#Elbow plot
plt.plot(range(1,6),wcss)
plt.title('Elbow Graph')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()


#Cluster Evaluation using Silhouette coefficient
from sklearn.metrics import silhouette_score

silhouette_coefficients = []

for k in range (2, 10):
    kmeans = KMeans(n_clusters = k, init = "random", random_state = 1)
    kmeans.fit(df_norm)
    score = silhouette_score(df_norm, kmeans.labels_)
    k = k
    Sil_coff = score
    silhouette_coefficients.append([k, Sil_coff])

silhouette_coefficients

#Clusters=2

'''
USING PCA
'''
from sklearn.decomposition import PCA

#defining pca model
pca=PCA(n_components=3)

#applying pca to normalized data set
df_pca=pd.DataFrame(pca.fit_transform(df_norm))


'''
PERFORMING KMEANS AND HEIRARCHICAL ON THE PCA DATAFRAME
'''

'''
HEIRARCHICAL
'''
#Creating Dendrogram
plt.figure(1, figsize = (16, 8))
dg=dendrogram(linkage(df_pca,method='complete'))


#creating clusters
clust_pca=AgglomerativeClustering(n_clusters=3,affinity='euclidean',linkage='ward')

y=clust_pca.fit_predict(df_norm)
clust_pca.labels_


#Clusters= 2


'''
USING KMEANS
'''

#Kmeans
pcss=[]
for i in range (1,6):
    kmeans=KMeans(n_clusters=i,random_state=2)
    kmeans.fit(df_pca)
    pcss.append(kmeans.inertia_)



#Elbow plot
plt.plot(range(1,6),pcss)
plt.title('Elbow Graph')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()



#Cluster Evaluation using Silhouette coefficient


silhouette_coefficients = []

for k in range (2, 10):
    kmeans = KMeans(n_clusters = k, init = "random", random_state = 1)
    kmeans.fit(df_pca)
    score = silhouette_score(df_pca, kmeans.labels_)
    k = k
    Sil_coff = score
    silhouette_coefficients.append([k, Sil_coff])

silhouette_coefficients

#Clusters= 2


'''
The results are similar with and without using PCA ie.Optimum number of clusters=2
'''

