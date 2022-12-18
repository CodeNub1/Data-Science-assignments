# -*- coding: utf-8 -*-
"""
Created on Fri Dec 16 06:31:07 2022

@author: Sidha
"""


import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import norm
from scipy.stats import chi2_contingency


#import dataset
df=pd.read_csv('C:/Users/Sidha/OneDrive/Desktop/Datasets/hypothesis/CustomerOrderform.csv')

df.isna().sum()
df=df.dropna()


df.Phillippines.value_counts()

df.Indonesia.value_counts()

df.Malta.value_counts()


df.India.value_counts()


# Make a contingency table
obs=np.array([[271,267,269,280],[29,33,31,20]])
obs


# Chi2 contengency independence test
chi2_contingency(obs) # (Chi2 stats value, p_value, df, expected obsvations)

