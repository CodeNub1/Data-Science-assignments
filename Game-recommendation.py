# -*- coding: utf-8 -*-
"""
Created on Fri Nov 25 18:21:42 2022

@author: Sidha
"""

import pandas as pd
import numpy as np

#importing data set
df=pd.read_csv('C:/Users/Sidha/OneDrive/Desktop/Datasets/Recommendation/game.csv')
df.columns
df.sort_values(by=['userId'])



#Check for missing values
df.isna().sum()


#number of unique users in the dataset
len(df['userId'].unique())

#number of unique game in the dataset
len(df['game'].unique())



#converting long data into wide data using pivot table
df1=df.pivot_table(index='userId',columns='game',values='rating').reset_index(drop=True)



#replacing the index values bu unique user ids
df1.index=df['userId'].unique()


#impute nan values with 0
df1.fillna(0,inplace=True)



#calculating cosine similarity between users
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cosine,correlation
x=1-pairwise_distances(df1.values,metric='cosine')


#nullifying diagonal values
np.fill_diagonal(x, 0)

#storing in datafrome format
y=pd.DataFrame(x)




#setting index and column names
y.index=df['userId'].unique()
y.columns=df['userId'].unique()





#most similar users
a=y.idxmax(axis=1)[0:]



#games played by users 
z=df[(df['userId']==39)|(df['userId']==1)]
z


user1=df[(df['userId']==39)]
user2=df[(df['userId']==1)]


user1['game']



#recommending user 1 the games of user 39
pd.merge(user1,user2,on='game',how='outer')




