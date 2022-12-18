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
df=pd.read_csv('C:/Users/Sidha/OneDrive/Desktop/Datasets/Neural networks/concrete.csv')




df.info()


#duplicates
df.duplicated().sum()

df=df.drop_duplicates()

df.isna().sum()


#normalizing
def norm_func(i):
    x = (i-i.min())/(i.max()-i.min())
    return (x)

df_norm=norm_func(df)




#input and output
x=df_norm.iloc[:,0:8]
y=df_norm['strength']


model = Sequential()
model.add(Dense(150, input_dim=8, activation='relu'))
model.add(Dense(200,activation='relu'))
model.add(Dense(1,activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# Fit the model
model.fit(x, y,epochs=500, batch_size=10)




accuracy = model.evaluate(x, y)
print('Accuracy:  ' %(accuracy*100))



