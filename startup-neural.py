# -*- coding: utf-8 -*-
"""
Created on Thu Dec 15 07:25:20 2022

@author: Sidha
"""
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from keras.utils import np_utils
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import preprocessing

#importing dataset
df=pd.read_csv('C:/Users/Sidha/OneDrive/Desktop/Datasets/Neural networks/50_Startups (2).csv')




df.info()


#duplicates
df.duplicated().sum()



df.isna().sum()

#only numerical columns
col='State'
df_num=df.loc[:, df.columns !=col]

#normalizing
def norm_func(i):
    x = (i-i.min())/(i.max()-i.min())
    return (x)

df_norm=norm_func(df_num)




#input and output
x=df_norm.iloc[:,0:3]
y=df_norm['Profit']


model = Sequential()
model.add(Dense(100, input_dim=3, activation='relu'))
model.add(Dense(50,activation='relu'))
model.add(Dense(1,activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# Fit the model
model.fit(x, y,epochs=150, batch_size=10)




accuracy = model.evaluate(x, y)
print('Accuracy:  ' %(accuracy*100))



