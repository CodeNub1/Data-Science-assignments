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
df=pd.read_csv('C:/Users/Sidha/OneDrive/Desktop/Datasets/Logosttic regression/Affairs.csv')


df.info()

df=df.drop('Unnamed: 0',axis=1)
      
#converting to binary     
df.naffairs[df.naffairs>0]=1
        
    

df.columns
#features and target

x = df.iloc[:,1:19]
y = df['naffairs']




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
(432+28)/(432+28+19+122)
             




