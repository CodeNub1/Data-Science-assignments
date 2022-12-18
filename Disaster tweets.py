
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



#importing data set
df=pd.read_csv('C:/Users/Sidha/OneDrive/Desktop/Datasets/Naive bayes/Disaster_tweets_NB.csv')


#dropping null columns
df=df.drop(columns=['id','keyword','location'])


from sklearn.model_selection import train_test_split
train, test = train_test_split(df, test_size = 0.2) 

train
test

#duplicates
train.duplicated().sum()
test.duplicated().sum()
train=train.drop_duplicates()
test=test.drop_duplicates()


#Separating independent and dependent variables

X_train = train.drop(columns=['target'])
Y_train = train['target']

X_test = test.drop(columns=['target'])
Y_test = test['target']




#vectorisation
from sklearn.feature_extraction.text import TfidfVectorizer

tf_idf_vect = TfidfVectorizer(max_features=300, sublinear_tf=True)
tf_idf_vect.fit(df['text'])



X_train_tfidf = tf_idf_vect.transform(X_train['text'])
X_test_tfidf = tf_idf_vect.transform(X_test['text'])



#naive bayes
from sklearn.naive_bayes import MultinomialNB
nb=MultinomialNB()
nb.fit(X_train_tfidf, Y_train)


#prediction
nb_validation_predictions = nb.predict(X_test_tfidf)
nb_training_predictions = nb.predict(X_train_tfidf)



#accuracy
from sklearn.metrics import accuracy_score
Validation=accuracy_score(nb_validation_predictions, Y_test) * 100
Training=accuracy_score(nb_training_predictions, Y_train) * 100

print(Validation,Training)








