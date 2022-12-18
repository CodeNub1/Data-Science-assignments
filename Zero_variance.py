# -*- coding: utf-8 -*-
"""
Created on Mon Nov  7 13:45:52 2022

@author: Sidha
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df=pd.read_csv('C:/Users/Sidha/OneDrive/Desktop/Datasets/Z_dataset.csv')


#creating a new data froame for numerical data

df_num=df.iloc[:, :-1]

#checking for zero or near zero variance columns

df.var()

#square breadth has a near zero variance so dropping it

df=df.drop('square.breadth',axis=1)


#rec.breadth has a near zero variance so dropping it
df=df.drop('rec.breadth',axis=1)



#square.length  has a near zero variance so dropping it
df=df.drop('square.length',axis=1)
df.var()

