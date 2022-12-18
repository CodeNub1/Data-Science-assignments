# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 17:13:21 2022

@author: Sidha
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from feature_engine.outliers import Winsorizer
import dtale


df=pd.read_csv('C:/Users/Sidha/OneDrive/Desktop/Datasets/boston_data.csv')

print(df)
df.info()


#boxplots

df.plot(kind='box',subplots=True,sharey=False,figsize=(10,6))

#using winsorizer to treat outliers
a=df.columns.values.tolist()
print(a)
a.remove('chas')
for i in a:
    winsor=Winsorizer(capping_method='iqr',tail='both',fold=1.5,variables=[i])
    df[i]=winsor.fit_transform(df[[i]])
    
#using different capping method for chas column
winsor=Winsorizer(capping_method='gaussian',tail='both',fold=3,variables=['chas'])
df['chas']=winsor.fit_transform(df[['chas']])


#boxplot after winsorization
df.plot(kind='box',subplots=True,sharey=False,figsize=(10,6))


