# -*- coding: utf-8 -*-
"""
Created on Sun Nov 27 06:57:06 2022

@author: Sidha
"""

import pandas as pd
import numpy as np
import joblib,difflib
from sklearn.feature_extraction.text import TfidfVectorizer# term frequency - inverse document frequncy is a numerical statistic that is intended to reflect how important a word is to document in a collecion or corpus
from sklearn.metrics.pairwise import cosine_similarity

#importing dataset
df=pd.read_csv('C:/Users/Sidha/OneDrive/Desktop/Datasets/Recommendation/Entertainment.csv')
df.head()
df.isna().sum()
df.columns



#selecting features
features=df['Category']

tfif=TfidfVectorizer()

feature_vector=tfif.fit_transform(features)
print(feature_vector)




#cosine similarity
sim=cosine_similarity(feature_vector)
sim.shape


#getting movie name input from user
movie_name=input('Enter any movie you have watched: ')


#creating a list with all the movie names
list_of_movies=df['Titles'].tolist()
print(list_of_movies)



#finding a close match for the movie name in the list
close_match=difflib.get_close_matches(movie_name,list_of_movies,cutoff=0.3)
print(close_match)



#finding the index of the movie with title
index_of_movie=df[df.Titles==close_match[0]].index.values[0]
print(index_of_movie)


# getting a list of similar movies

similarity_score = list(enumerate(sim[index_of_movie]))
print(similarity_score)


#sorting movies based on similarity score
sorted_similar=sorted(similarity_score,key=lambda x:x[1],reverse = True)
print(sorted_similar)



#print name of movies based on index of input given
print('Movies suggested for you : \n')

i = 1

for movie in sorted_similar:
  index = movie[0]
  title_of_movie=df[df.index==index]['Titles'].values[0]
  if (i<30):
    print(i, '.',title_of_movie)
    i+=1