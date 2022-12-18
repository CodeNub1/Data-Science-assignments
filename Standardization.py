# -*- coding: utf-8 -*-
"""
Created on Mon Nov  7 17:29:06 2022

@author: Sidha
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


df=pd.read_csv('C:/Users/Sidha/OneDrive/Desktop/Datasets/Seeds_data.csv')



# Initialise the Scaler
scaler = StandardScaler()

# To scale data
df = scaler.fit_transform(df)


type(df)

#converting it back to data frame

df=pd.DataFrame(df)

type(df)

