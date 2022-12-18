# -*- coding: utf-8 -*-
"""
Created on Fri Dec 16 06:45:08 2022

@author: Sidha
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import norm
from scipy.stats import chi2_contingency


df=pd.read_csv('C:/Users/Sidha/OneDrive/Desktop/Datasets/hypothesis/BuyerRatio.csv')


# Make dimensional array
obs=np.array([[50,142,131,70],[435,1523,1356,750]])
obs


# Chi2 contengency independence test
chi2_contingency(obs) # o/p is (Chi2 stats value, p_value, df, expected obsvations)











