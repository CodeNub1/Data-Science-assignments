# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 18:39:40 2022

@author: Sidha
"""

import pandas as pd
import  numpy as np
import matplotlib.pyplot as plt
from collections import Counter
df=pd.read_excel(r'C:/Users/Sidha/OneDrive/Desktop/Datasets/Assignment_module02 (1).xlsx')

df.describe()
#MEAN
df.mean()
#MEDIAN
df.median()
#MODE
df.mode()
#VARIANCE
df.var()
#STANDARD DEVIATION
df.std()
#RANGE
print('Range\n',df.max()-df.min())







a=[24.33,25.53,25.41,24.14,29.62,28.25,25.81,24.39,40.26,32.95,91.36,25.99,39.42,26.71,35.00]


plt.hist(a, edgecolor='black')
plt.ylabel('Measure')

np.mean(a)
np.median(a)
np.var(a)
np.std(a)

