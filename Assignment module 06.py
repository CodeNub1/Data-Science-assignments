# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 21:59:33 2022

@author: Sidha
"""

import pandas as pd
import matplotlib.pyplot as mp
import numpy as np

df=pd.read_csv('C:/Users/Sidha/Downloads/Indian_cities (2)/Indian_cities.csv')


type(df)
#1a
state_sex_ratio=df[['state_name','sex_ratio']].sort_values('sex_ratio',ascending=False).head(10)

print(state_sex_ratio)


#1b
for col in df.columns:
    print(col)
city_graduates=df[['name_of_city','total_graduates']].sort_values('total_graduates',ascending=False).head(10)
print(city_graduates)

#1c
city_location_literacy=df[['name_of_city','location','effective_literacy_rate_total']].sort_values('effective_literacy_rate_total',ascending=False).head(10)
print(city_location_literacy)


#2a
x=df['literates_total']

mp.hist(x,edgecolor='black')
mp.ylabel('Total Literates')
print(df.shape)
#inference- 
#1.right skewed data-
#2 outliers towards the right

#2b
x=df['male_graduates']
y=df['female_graduates']
colors=np.array(['red','green'])
mp.scatter(x, y,c='green',edgecolor='red')
mp.ylabel('Female Graduates')
mp.xlabel("male graduates")





#3a
x=df['effective_literacy_rate_total']
mp.boxplot(x,vert=False)

#inference-
#1-ouliers towards the left
#2-left skewed



#3b
#duplicate detection
df.duplicated().sum()
#delete duplicates
df.drop_duplicates()

print(df)
