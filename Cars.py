# -*- coding: utf-8 -*-
"""
Created on Fri Dec 16 11:54:19 2022

@author: Sidha
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import norm


cars=pd.read_csv('C:/Users/Sidha/OneDrive/Desktop/Datasets/Cars.csv')


sns.boxplot(cars.MPG)



# P(MPG>38)
1-stats.norm.cdf(38,cars.MPG.mean(),cars.MPG.std())


# P(MPG<40)
stats.norm.cdf(40,cars.MPG.mean(),cars.MPG.std())


# P (20)
stats.norm.cdf(0.50,cars.MPG.mean(),cars.MPG.std())-stats.norm.cdf(0.20,cars.MPG.mean(),cars.MPG.std()) 