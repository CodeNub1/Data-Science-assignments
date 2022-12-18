# -*- coding: utf-8 -*-
"""
Created on Fri Dec  9 06:22:28 2022

@author: Sidha
"""

'''
Decision tree
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


#importing data set
df=pd.read_csv('C:/Users/Sidha/OneDrive/Desktop/Datasets/Knn and decision tree/Diabetes.csv')

df.columns
df.info()



#dummy variable creation
df=pd.get_dummies(df,columns=[' Class variable'],drop_first=True)

#feature variables
x=df.drop([' Class variable_YES'], axis=1)
x
  
#target variable
y=df[' Class variable_YES']
y




from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)



#decision tree

model = DecisionTreeClassifier()



#train decision tree
model = model.fit(x_train,y_train)

#Predict the response for test dataset
y_pred = model.predict(x_test)



#Evaluation using Accuracy score
from sklearn import metrics
print("Accuracy:",metrics.accuracy_score(y_test, y_pred)*100)



#Evaluation using Confusion matrix
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,y_pred)


#tree plot
from sklearn import tree
tree.plot_tree(model);




'''
Random forest
'''


from sklearn.ensemble import RandomForestClassifier
model= RandomForestClassifier(n_estimators=200,max_depth=20,min_samples_split=40,criterion='gini')


model.fit(x_train,y_train)

#accuracy
print(model.score(x_train, y_train))



#accuray
print(model.score(x_test, y_test))
