# -*- coding: utf-8 -*-
"""
Created on Sun Dec 11 09:49:58 2022

@author: Sidha
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


#importing dataset
df=pd.read_csv('C:/Users/Sidha/OneDrive/Desktop/Datasets/KNN/glass.csv')

df.duplicated().sum()

df=df.drop_duplicates()




#pair plot
sns.pairplot(df,hue='Type')
plt.show()




# Normalization function 
def norm_func(i):
    x = (i-i.min())	/ (i.max()-i.min())
    return (x)


df_norm = norm_func(df.iloc[:, 0:9])



#separating into features and targer
x=np.array(df_norm.iloc[:,:])#feature
y=np.array(df['Type'])#target



#splitting into train ans test data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)





#Knn classifier

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 2)

KNN = knn.fit(x_train, y_train)  # Train the kNN model



#evaluate the model
pred_train = knn.predict(x_train)



# Cross table
pd.crosstab(y_train, pred_train, rownames = ['Actual'], colnames = ['Predictions']) 





#accuracy score
from sklearn.metrics import accuracy_score
print(accuracy_score(y_train, pred_train)) 







# Evaluate the model with test data
pred_test=knn.predict(x_test)
print(accuracy_score(y_test, pred_test))
pd.crosstab(y_test, pred_test, rownames = ['Actual'], colnames= ['Predictions']) 





#accuracy score
print(accuracy_score(y_test, pred_test)) 



knn.score(x_train,y_train)



#classification report
from sklearn.metrics import accuracy_score, classification_report


print(classification_report(y_test,pred_test))



























