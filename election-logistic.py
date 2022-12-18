# -*- coding: utf-8 -*-
"""
Created on Sun Dec 11 17:02:29 2022

@author: Sidha
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



#importing dataset
df=pd.read_csv('C:/Users/Sidha/OneDrive/Desktop/Datasets/Logosttic regression/election_data.csv')


df.info()

df.isna().sum()
    
df=df.dropna()  



 

df.columns
#features and target

x = df[['Amount Spent','Popularity Rank']]
y = df['Result']




# Logistic regression model
from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression()
classifier.fit(x,y)






# Predict for x dataset
y_pred=classifier.predict(x)
y_pred




#model accuracy
# Confusion Matrix for the model accuracy
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y,y_pred)
confusion_matrix


#accuracy
(3+5)/(3+5+1+1)
             




