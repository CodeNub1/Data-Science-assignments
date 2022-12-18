# -*- coding: utf-8 -*-
"""
Created on Fri Dec 16 07:52:28 2022

@author: Sidha
"""

import pandas as pd 
import numpy as np 
import seaborn as sns



#importing dataset
df=pd.read_excel('C:/Users/Sidha/OneDrive/Desktop/Datasets/Ensemble/Ensemble_Password_Strength.xlsx')
df.columns


df.info()

#converting into object
df['characters'] = df['characters'].astype(str)

from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()



df['characters'] = labelencoder.fit_transform(df['characters'])




#feature and target

feature = df['characters']
target = df['characters_strength']




'''
Bagging
'''
# Splitting data into training and testing data set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(feature,target, test_size = 0.2,random_state=7)

y_train=y_train.reshape(-1,1)

from sklearn import tree 
clftree = tree.DecisionTreeClassifier()
from sklearn.ensemble import BaggingClassifier


bag_clf = BaggingClassifier(base_estimator = clftree, n_estimators =500,bootstrap = True, n_jobs = 1, random_state = 77)

bag_clf.fit(x_train, y_train)



from sklearn.metrics import accuracy_score, confusion_matrix


# Evaluation on Testing Data
confusion_matrix(y_test, bag_clf.predict(x_test))
accuracy_score(y_test, bag_clf.predict(x_test))

# Evaluation on Training Data
confusion_matrix(y_train, bag_clf.predict(x_train))
accuracy_score(y_train, bag_clf.predict(x_train))


'''
Boosting
'''
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier

# Splitting data into training and testing data set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(feature,target, test_size = 0.2,random_state=7)
                                            

#decision tree
dt = DecisionTreeClassifier() #storing the classifer in dt

dt.fit(x_train, y_train) #fitting te model 

dt.score(x_test, y_test) #checking the score like accuracy

dt.score(x_train, y_train)
# model is overfitting 

# Ada boosting 
ada = AdaBoostClassifier(DecisionTreeClassifier(), n_estimators=10, learning_rate=7)
ada.fit(x_train,y_train)

ada.score(x_test,y_test)

ada.score(x_train,y_train)
       

'''

Voting
'''
                                   
# Splitting data into training and testing data set

x_train, x_test, y_train, y_test = train_test_split(feature,target, test_size = 0.2,random_state=7)
                                          
from sklearn.ensemble import VotingClassifier
# Voting Classifier 
from sklearn.linear_model import LogisticRegression # importing logistc regression
from sklearn.svm import SVC # importing Svm 

lr = LogisticRegression() 
dt = DecisionTreeClassifier()
svm = SVC(kernel= 'poly', degree=2)

evc = VotingClassifier(estimators=[('lr', lr),('dt', dt),('svm', svm)], voting='hard')

evc.fit(x_train, y_train)

evc.score(x_test, y_test)

evc.score(x_train, y_train)





