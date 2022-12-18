# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 17:51:01 2022

@author: Sidha
"""

import pandas as pd
import numpy as np


#imprting worksheet
df=pd.read_csv('C:/Users/Sidha/OneDrive/Desktop/Datasets/Clustering/crime_data.csv')
df.info()


#dropping Unnamed  column with countries into a new dataframe
df1=df.drop(['Unnamed: 0'],axis=1)
df1.info()


#Normalizing the data using standard scaler
from sklearn.preprocessing import StandardScaler
df1_norm=StandardScaler().fit_transform(df1)
df1_norm



#Using DBscan clustering
from sklearn.cluster import DBSCAN
dbscan=DBSCAN(eps=1,min_samples=4)
dbscan.fit(df1_norm)
dbscan.labels_



#adding cluster to original data set
df['clusters'] = dbscan.labels_





#sorting the data set according to clusters
df.groupby('clusters').agg(['mean']).reset_index()
