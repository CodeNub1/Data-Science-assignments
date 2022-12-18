# -*- coding: utf-8 -*-
"""
Created on Thu Dec 15 12:55:17 2022

@author: Sidha
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import norm


#import dataset
df=pd.read_csv('C:/Users/Sidha/OneDrive/Desktop/Datasets/hypothesis/lab_tat_updated.csv')



# Anova ftest statistics: 
p_value=stats.f_oneway(df.iloc[:,0],df.iloc[:,1],df.iloc[:,2],df.iloc[:,3])
p_value


p_value[1]



