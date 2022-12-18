# -*- coding: utf-8 -*-
"""
Created on Tue Dec  6 16:38:43 2022

@author: Sidha
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


#importing data set
train=pd.read_csv('C:/Users/Sidha/OneDrive/Desktop/Datasets/Naive bayes/SalaryData_Train.csv')
test=pd.read_csv('C:/Users/Sidha/OneDrive/Desktop/Datasets/Naive bayes/SalaryData_Test.csv')


#checking for duplicates
train.duplicated().sum()
test.duplicated().sum()


#dropping duplicates
train=train.drop_duplicates()
test=test.drop_duplicates()



#pre processing on categorical data
from sklearn.preprocessing import LabelEncoder

columns=["workclass","education","maritalstatus","occupation","relationship","race","sex","native"]
number = LabelEncoder()
for i in columns:
        train[i]= number.fit_transform(train[i])
        test[i]=number.fit_transform(test[i])
        
        
#storing data in array
colnames=train.columns
x_train = train[colnames[0:13]].values
y_train = train[colnames[13]].values
x_test = test[colnames[0:13]].values
y_test = test[colnames[13]].values
        
        

#normalizing
def norm_func(i):
    x = (i-i.min())/(i.max()-i.min())
    return (x)
   
x_train = norm_func(x_train)
x_test =  norm_func(x_test)





#applying naive bayes for classification
from sklearn.naive_bayes import MultinomialNB as MB

M_model=MB()
train_pred_multi=M_model.fit(x_train,y_train).predict(x_train)
test_pred_multi=M_model.fit(x_train,y_train).predict(x_test)

#train accuracy
train_acc_multi=np.mean(train_pred_multi==y_train)
train_acc_multi

#test accuracy
test_acc_multi=np.mean(test_pred_multi==y_test)
test_acc_multi




#Confusion matrix
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, test_pred_multi)
confusion_matrix





#calculating the accuracy of this model w.r.t. this dataset
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,test_pred_multi))






#gaussian naive bayes
from sklearn.naive_bayes import GaussianNB as GB
G_model=GB()
train_pred_gau=G_model.fit(x_train,y_train).predict(x_train)
test_pred_gau=G_model.fit(x_train,y_train).predict(x_test)


#train accuracy
train_acc_gau=np.mean(train_pred_gau==y_train)
train_acc_gau


#test accuracy
test_acc_gau=np.mean(test_pred_gau==y_test)
test_acc_gau




#confusion matrix
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, test_pred_gau)

confusion_matrix


#calculating the accuracy of this model w.r.t. this dataset
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,test_pred_gau))




