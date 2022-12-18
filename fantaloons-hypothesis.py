# -*- coding: utf-8 -*-
"""
Created on Fri Dec 16 06:40:57 2022

@author: Sidha
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import norm
from scipy.stats import chi2_contingency


#import dataset
df=pd.read_csv('C:/Users/Sidha/OneDrive/Desktop/Datasets/hypothesis/Fantaloons.csv')

df.isna().sum()
df=df.dropna()


df.Weekdays.value_counts()

df.Weekend.value_counts()




# Make a contingency table
obs=np.array([[287,233],[113,167]])
obs


# Chi2 contengency independence test
chi2_contingency(obs) # (Chi2 stats value, p_value, df, expected obsvations)

