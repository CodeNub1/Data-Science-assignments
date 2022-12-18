# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 17:51:01 2022

@author: Sidha
"""

import pandas as pd
import matplotlib.pyplot as plt



#reading specific sheet from worksheet
df=pd.read_csv('C:/Users/Sidha/OneDrive/Desktop/Datasets/Clustering/crime_data.csv')
df.info()



#dropping ID column 
df1=df.drop(['Unnamed: 0'])
df1.info()
df1.shape

#Normalizing the data using standard scaler
from sklearn.preprocessing import StandardScaler
df1_norm=StandardScaler().fit_transform(df1)
df1_norm





#using elbow plot to find optimim number of clusters-K value
from sklearn.cluster import KMeans
cwss=[]
k=range(1,5)

for i in k:
    kmeans=KMeans(n_clusters=i)
    kmeans.fit(df1_norm)
    cwss.append(kmeans.inertia_)


cwss



#creating elbow curve
scalex=[0,1,2,3,4,5]
scaley=range(100,300)
plt.plot(k,cwss)
plt.title('Elbow Graph')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()




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

