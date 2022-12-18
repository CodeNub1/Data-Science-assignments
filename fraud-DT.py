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
df=pd.read_csv('C:/Users/Sidha/OneDrive/Desktop/Datasets/Knn and decision tree/Fraud_check.csv')


#dummy variable creation
df=pd.get_dummies(df,columns=['Undergrad','Marital.Status','Urban'],drop_first=True)

#creating a column for tax income
#assuming tax less than 30000 as risky and others as good
df["TaxInc"] = pd.cut(df["Taxable.Income"], bins = [10002,30000,99620], labels = ["Risky", "Good"])

#dummy for nre column
df = pd.get_dummies(df,columns = ["TaxInc"],drop_first=True)



# Normalization of data 
def norm_func(i):
    x = (i-i.min())/(i.max()-i.min())
    return (x)


df_norm = norm_func(df.iloc[:,1:])

#declaring features and target
x = df_norm.drop(['TaxInc_Good'], axis=1)
y = df_norm['TaxInc_Good']


from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=0)



##Converting the Taxable income to bucketing. 
df_norm["income"]="<=30000"
df_norm.loc[df["Taxable.Income"]>=30000,"income"]="Good"
df_norm.loc[df["Taxable.Income"]<=30000,"income"]="Risky"


##Droping the Taxable income variable
df.drop(["Taxable.Income"],axis=1,inplace=True)

#renaming columns to remove errors
df.rename(columns={"Undergrad":"undergrad","Marital.Status":"marital","City.Population":"population","Work.Experience":"experience","Urban":"urban"},inplace=True)


from sklearn import preprocessing
le=preprocessing.LabelEncoder()
for column_name in df.columns:
    if df[column_name].dtype == object:
        df[column_name] = le.fit_transform(df[column_name])
    else:
        pass


##Splitting the data into featuers and labels
features = df.iloc[:,0:5]
labels = df.iloc[:,5]



## Collecting the column names
colnames = list(df.columns)
predictors = colnames[0:5]
target = colnames[5]
##Splitting the data into train and test


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(features,labels,test_size = 0.2,stratify = labels)




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

fraud=pd.read_csv('C:/Users/Sidha/OneDrive/Desktop/Datasets/Knn and decision tree/Fraud_check.csv')


#Creating dummy vairables for ['Undergrad','Marital.Status','Urban'] dropping first dummy variable
fraud=pd.get_dummies(fraud,columns=['Undergrad','Marital.Status','Urban'], drop_first=True)


#Creating new cols TaxInc and dividing 'Taxable.Income' cols on the basis of [0,30000,99620] for Risky and Good
fraud["TaxInc"] = pd.cut(fraud["Taxable.Income"], bins = [0,30000,99620], labels = ["Risky", "Good"])



#After creation of new col. TaxInc also made its dummies var concating right side of fraud
fraud = pd.get_dummies(fraud,columns = ["TaxInc"],drop_first=True)



X = fraud.iloc[:,1:7]
Y = fraud.iloc[:,-1]


# Splitting data into train & test
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.3, random_state=0)


from sklearn.ensemble import RandomForestClassifier
model= RandomForestClassifier(n_estimators=200,max_depth=20,min_samples_split=40,criterion='gini')


model.fit(Xtrain,Ytrain)

#accuracy
print(model.score(Xtrain, Ytrain))



#accuray
print(model.score(Xtest, Ytest))
