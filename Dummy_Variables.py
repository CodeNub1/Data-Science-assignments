# -*- coding: utf-8 -*-
"""
Created on Mon Nov  7 12:04:44 2022

@author: Sidha
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder 


df=pd.read_csv('C:/Users/Sidha/OneDrive/Desktop/Datasets/animal_category.csv')


df.info()


#separating categorical values to anew dataframe
df_cat=df.iloc[:,1:]
df_cat


df_cat=pd.get_dummies(df_cat,drop_first=True)


