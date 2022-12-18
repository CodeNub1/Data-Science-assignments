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
df=pd.read_csv('C:/Users/Sidha/OneDrive/Desktop/Datasets/Knn and decision tree/Company_Data.csv')


#dummy variable creation
df=pd.get_dummies(df,columns=['Urban','US'],drop_first=True)



from sklearn.model_selection import train_test_split


df['ShelveLoc']=df['ShelveLoc'].map({'Good':1,'Medium':2,'Bad':3})



colnames=list(df.columns)

#declaring feature and target
x=df.iloc[:,0:6]
y=df['ShelveLoc']


#splitting into train and test
x_train, x_test,y_train,y_test = train_test_split(x,y, test_size=0.2,random_state=40)



#building decision tree using Entropy
from sklearn.tree import DecisionTreeClassifier as Ds
from sklearn.tree import plot_tree 

model=Ds(criterion='entropy',max_depth=3)
model.fit(x_train,y_train)


#tree plot
from sklearn import tree
tree.plot_tree(model);


#prediction on test data
preds=model.predict(x_test)
pd.Series(preds).value_counts()



#accuracy
np.mean(preds==y_test)


'''
Random forest
'''

# Labels are the values we want to predict
labels = np.array(df['Income'])



# axis 1 refers to the columns
features= df.drop('Income', axis = 1)



feature_list = list(df.columns)
features = np.array(df)

#splitting data
train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.25, random_state=42)


#baseline-hitorical predictions
baseline_preds = test_features[:, feature_list.index('Sales')]


# Baseline errors, and display average baseline error
baseline_errors = abs(baseline_preds - test_labels)
print(round(np.mean(baseline_errors), 2))


from sklearn.ensemble import RandomForestRegressor
#forest model with 500 decision trees
rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)


#train model on training data
rf.fit(train_features, train_labels)



#prediction using forest medthod
predictions = rf.predict(test_features)


# Calculate the absolute errors
errors = abs(predictions - test_labels)

print(round(np.mean(errors), 2), 'degrees.')


# Calculate mean absolute percentage error (MAPE)
mape = 100 * (errors / test_labels)
# accuracy
accuracy = 100 - np.mean(mape)
print( round(accuracy, 2))

