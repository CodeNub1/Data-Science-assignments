# -*- coding: utf-8 -*-
"""
Created on Tue Dec 13 10:04:47 2022

@author: Sidha
"""

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 


#importing dataset
df=pd.read_csv('C:/Users/Sidha/OneDrive/Desktop/Datasets/Lasso and ridge/RetailPrices_data.csv')
df.columns

#categorical to numerical
df=pd.get_dummies(df,columns=['cd','multi','premium'],drop_first=True)



#separating to features and terget
x=df.iloc[:,2:]
y=df['price']


#splitting data into train and test
from sklearn.model_selection import train_test_split 

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.25)

#apply Lasso regression
from sklearn.linear_model import Lasso 

# Train the model 
lasso = Lasso(alpha = 0.05) 
lasso.fit(x_train, y_train) 
y_pred1 = lasso.predict(x_test)


# Calculate Mean Squared Error 
mean_squared_error = np.mean((y_pred1 - y_test)**2) 
print("Mean squared error on test set", mean_squared_error) 
lasso_coeff = pd.DataFrame() 
lasso_coeff["Columns"] = x_train.columns 
lasso_coeff['Coefficient Estimate'] = pd.Series(lasso.coef_) 

print(lasso_coeff) 



#ridge regression
from sklearn.linear_model import Ridge 

# Train the model 
ridgeR = Ridge(alpha = .99) 
ridgeR.fit(x_train, y_train) 
y_pred = ridgeR.predict(x_test) 

# calculate mean square error 
mean_squared_error_ridge = np.mean((y_pred - y_test)**2) 
print(mean_squared_error_ridge) 

# get ridge coefficient and print them 
ridge_coefficient = pd.DataFrame() 
ridge_coefficient["Columns"]= x_train.columns 
ridge_coefficient['Coefficient Estimate'] = pd.Series(ridgeR.coef_) 
print(ridge_coefficient) 
























































