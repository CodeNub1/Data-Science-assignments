# -*- coding: utf-8 -*-
"""
Created on Sun Dec 11 12:26:31 2022

@author: Sidha
"""


import pandas as pd 
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt


#importing dataset
Test=pd.read_csv('C:/Users/Sidha/OneDrive/Desktop/Datasets/SVM/SalaryData_Test.csv')
Train=pd.read_csv('C:/Users/Sidha/OneDrive/Desktop/Datasets/SVM/SalaryData_Train.csv')


#pre-processing the categoriacl columns

from sklearn.preprocessing import LabelEncoder

string_columns = ["workclass","education","maritalstatus","occupation","relationship","race","sex","native"]


num = LabelEncoder()
for i in string_columns:
        Train[i]= num.fit_transform(Train[i])
        Test[i]=num.fit_transform(Test[i])



#Normalmization
def norm_func(i):
    x = (i-i.min())/(i.max()-i.min())
    return (x)


#column names
colnames=Train.columns



#features and target
x_train = Train[colnames[0:13]]
y_train = Train[colnames[13]]
x_test = Test[colnames[0:13]]
y_test = Test[colnames[13]]


#normalizing

x_train = norm_func(x_train)
x_test =  norm_func(x_test)



#SVM
from sklearn.svm import SVC
# kernel = linear
model_linear = SVC(kernel = "linear")
model_linear.fit(x_train,y_train)




pred_test_linear = model_linear.predict(x_test)





#accuracy
np.mean(pred_test_linear==y_test)







# kernel = rbf
model_rbf = SVC(kernel = "rbf")
model_rbf.fit(x_train,y_train)
pred_test_rbf = model_rbf.predict(x_test)



#accuracy
np.mean(pred_test_rbf==y_test)






'''
Forest
'''

import pandas as pd 
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt


#importing dataset
df=pd.read_csv('C:/Users/Sidha/OneDrive/Desktop/Datasets/SVM/forestfires.csv')
df.columns


#features and target
feature=df.iloc[:,2:30]
target=df['size_category']




#Normalmization
def norm_func(i):
    x = (i-i.min())/(i.max()-i.min())
    return (x)


#normalizing

feature_norm = norm_func(feature)

#splitting data
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(feature_norm,target,test_size = 0.35, stratify = target)



#SVM
from sklearn.svm import SVC
# kernel = linear
model_linear = SVC(kernel = "linear")
model_linear.fit(x_train,y_train)




pred_test_linear = model_linear.predict(x_test)





#accuracy
print(np.mean(pred_test_linear==y_test))








































