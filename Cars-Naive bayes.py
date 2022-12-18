
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



#importing data set
df=pd.read_csv('C:/Users/Sidha/OneDrive/Desktop/Datasets/Naive bayes/NB_Car_Ad.csv')
df=df.drop('User ID',axis=1)

from sklearn.model_selection import train_test_split
train, test = train_test_split(df, test_size = 0.2) 

train
test

#duplicates
train.duplicated().sum()
test.duplicated().sum()
train=train.drop_duplicates()
test=test.drop_duplicates()


#pre processing on categorical data
from sklearn.preprocessing import LabelEncoder


number = LabelEncoder()
train['Gender']= number.fit_transform(train['Gender'])
test['Gender']=number.fit_transform(test['Gender'])
        
        
#storing data in array
colnames=train.columns
colnames
x_train = train[colnames[0:4]].values
y_train = train[colnames[3]].values
x_test = test[colnames[0:4]].values
y_test = test[colnames[3]].values
        
        

#normalizing
def norm_func(i):
    x = (i-i.min())/(i.max()-i.min())
    return (x)
   
x_train = norm_func(x_train)
x_test =  norm_func(x_test)





#applying naive bayes for classification
from sklearn.naive_bayes import BernoulliNB as BB

M_model=BB()
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










