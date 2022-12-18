# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 18:43:25 2022

@author: Sidha
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df=pd.read_csv('C:/Users/Sidha/OneDrive/Desktop/Datasets/Q1_a.csv')
df.describe()         
df.info()
df.mode()

df.var()
#Q1
df.describe()
'''
Mean speed=15.4
median speed=15
mode=25
std=5.89
var=27.9

Mean distance=15.4
median distance=15
mode=25
std=25.7
var=664.1
'''
#skewness
df.skew()
'''
Index    0.000000
speed   -0.117510
dist     0.806895

'''
#inference
'''
Speed data is not normal
It has a slight negative skew of -0.117510
Mean<Median<Mode

Distance data is not normal
It has a slight positve skew of 0.806
Mean>Median>Mode
'''
#kutosis
df.kurt()
'''
Index   -1.200000
speed   -0.508994
dist     0.405053
'''
#inference
'''
Speed data is platykurtic
Distance data is leptokurtic
'''


#plot
x=df['speed']
plt.hist(x,edgecolor='black')


y=df['dist']
plt.hist(y,edgecolor='black')




#Q2
df2=pd.read_csv('C:/Users/Sidha/OneDrive/Desktop/Datasets/Q2_b.csv')
df2.info()

df2.describe()
df2.skew()
df2.kurt()
df2.var()
df2.mode()

x=df2['SP']
plt.hist(x,edgecolor='black')
plt.xlabel('SP')

y=df2['WT']
plt.hist(y,edgecolor='black')
plt.xlabel('WT')

df2.mode()
#Q3
def mid(array):
    if len(array)%2==0:
        y=int(len(array)/2)
        x=(array[y]+array[y-1])/2
    else:
        y=int((len(array)-1)/2)
        x=array[y]
    return x

a=np.array([34,36,36,38,38,39,39,40,40,41,41,41,41,42,42,45,49,56])

a.mean()
a.max()
a.var()
a.std()
a.mode()
mid(a)

plt.hist(a,edgecolor='black')

plt.boxplot(a,vert=False)

