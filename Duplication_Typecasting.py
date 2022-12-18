# -*- coding: utf-8 -*-
"""
Created on Tue Nov  1 18:08:49 2022

@author: Sidha
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing


'''
Importing the data online retail.csv
'''
df=pd.read_csv('C:/Users/Sidha/OneDrive/Desktop/Datasets/OnlineRetail.csv',encoding='unicode_escape')

df.info()

'''
The dataset contains CustomerID which is being interpreted as Integer by Python. 
CustomerID is a unique number given to each customer. 
Hence it should be treated as a categorical data. 
We can convert the integer data to string type.
'''

df.CustomerID=df.CustomerID.astype('str')

df.dtypes



#Q2
'''
Checking duplicates
'''
duplicate=df.duplicated()
duplicate.sum()

'''
there are total of 5268 duplicate values-dropping them from dataframe
'''
df=df.drop_duplicates()

duplicate=df.duplicated().sum()
print(duplicate)

'''
Now there are no duplicates
'''



#Q3
'''
EDA
1st moment of business decision
'''
print('Mean\n')
print(df[['Quantity','UnitPrice']].mean())
print('\n')
print('Median\n')
print(df[['Quantity','UnitPrice']].median())
print('\n')
print('Mode\n')
print(df[['Quantity','UnitPrice']].mode())

'''
2nd moment of business decision
'''

df1=df[['Quantity','UnitPrice']]
print('Variance\n')
print(df1.var())
print('Standard Deviation\n')
print(df1.std())

'''
3rd moment of business decision
'''
df1.skew()

'''
4th moment of business decision
'''

df1.kurt()


#Histogram

quant=abs(df['Quantity'])
unit=abs(df['UnitPrice'])

plt.hist(quant)
plt.hist(unit)




#Boxplot

plt.boxplot(quant,vert=False)


plt.boxplot(unit,vert=False)



#scatter

plt.scatter(unit,quant)
