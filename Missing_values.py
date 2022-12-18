# -*- coding: utf-8 -*-
"""
Created on Sun Nov  6 19:57:29 2022

@author: Sidha
"""
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
from feature_engine.imputation import RandomSampleImputer



df=pd.read_csv('C:/Users/Sidha/OneDrive/Desktop/Datasets/claimants.csv')


#checking missing values
df.isnull().sum()

#checking th plots for imputation strategy
df.plot(kind='box',subplots=True,sharey=False,figsize = (10, 6))


#using random imputer in CLMSEX

random=RandomSampleImputer(['CLMSEX'])
df['CLMSEX']=pd.DataFrame(random.fit_transform(df[['CLMSEX']]))
df['CLMSEX'].isnull().sum()


#using random imputer in CLMINSUR
random=RandomSampleImputer(['CLMINSUR'])
df['CLMINSUR']=pd.DataFrame(random.fit_transform(df[['CLMINSUR']]))
df['CLMINSUR'].isnull().sum()


#using random imputer in SEATBELT
random=RandomSampleImputer(['SEATBELT'])
df['SEATBELT']=pd.DataFrame(random.fit_transform(df[['SEATBELT']]))
df['SEATBELT'].isnull().sum()




#using median imputer in CLMAGE 
median_imputer = SimpleImputer(missing_values = 28.4144, strategy = 'median')
df["CLMAGE"] = pd.DataFrame(median_imputer.fit_transform(df[["CLMAGE"]]))
df["CLMAGE"].isna().sum()





