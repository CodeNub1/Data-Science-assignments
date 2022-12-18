# -*- coding: utf-8 -*-
"""
Created on Thu Dec 15 13:02:52 2022

@author: Sidha
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import norm


#import dataset
df=pd.read_csv('C:/Users/Sidha/OneDrive/Desktop/Datasets/hypothesis/Cutlets.csv')

df.isna().sum()
df=df.dropna()




unita=pd.Series(df.iloc[:,0])
unitb=pd.Series(df.iloc[:,1])

#ttest
p_value=stats.ttest_ind(unita,unitb)
p_value


p_value[1]

