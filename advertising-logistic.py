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
df=pd.read_csv('C:/Users/Sidha/OneDrive/Desktop/Datasets/Logosttic regression/advertising.csv')


df.info()




#converting date time string to object
df['Timestamp'] = pd.to_datetime(df['Timestamp']) 


df['Month'] = df['Timestamp'].dt.month 
# Creates a new column called Month
df['Day'] = df['Timestamp'].dt.day     
# Creates a new column called Day
df['Hour'] = df['Timestamp'].dt.hour   
# Creates a new column called Hour
df["Weekday"] = df['Timestamp'].dt.dayofweek 
# Dropping timestamp column to avoid redundancy
df = df.drop(['Timestamp'], axis=1) # deleting timestamp




df.columns
#features and target

x = df[['Daily_Time_ Spent _on_Site', 'Age', 'Area_Income','Daily Internet Usage', 'Male']]
y = df['Clicked_on_Ad']




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
(464+433)/(464+36+67+433)
             




