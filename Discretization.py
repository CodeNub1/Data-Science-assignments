# -*- coding: utf-8 -*-
"""
Created on Mon Nov  7 14:58:32 2022

@author: Sidha
"""

import pandas as pd
import numpy as np

df=pd.read_csv('C:/Users/Sidha/OneDrive/Desktop/Datasets/iris.csv')

df.info()

df.describe()

#using adaptive binning to convert numerical columns into discrete feature
labels_sp=['short','medium','long']
df['Sepal.Length_new'] = pd.qcut(df['Sepal.Length'], q=3, labels=labels_sp)


df['Sepal.Width_new'] = pd.qcut(df['Sepal.Width'], q=3, labels=labels_sp)


df['Petal.Length_new'] = pd.qcut(df['Petal.Length'], q=3, labels=labels_sp)


df['Petal.Width_new'] = pd.qcut(df['Petal.Width'], q=3, labels=labels_sp)

