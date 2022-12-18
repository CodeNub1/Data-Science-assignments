# -*- coding: utf-8 -*-
"""
Created on Mon Nov  7 17:38:26 2022

@author: Sidha
"""
import pandas as pd
from feature_engine import transformation
import numpy as np
import scipy.stats as stats
from scipy import stats
import pylab

df=pd.read_csv('C:/Users/Sidha/OneDrive/Desktop/Datasets/calories_consumed.csv')

df.head()
df.info()

#checking whether data is normal
stats.probplot(df['Weight gained (grams)'], dist="norm", plot=pylab)


stats.probplot(df['Calories Consumed'], dist="norm", plot=pylab)


'''
Calories consumed is nearly normal
weight gained is not normal



Using log function to tranform
'''

stats.probplot(np.log(df['Weight gained (grams)']), dist = "norm", plot = pylab)


#using sqrt tranformation
stats.probplot(np.sqrt(df['Weight gained (grams)']), dist = "norm", plot = pylab)


#using square tranformation

stats.probplot(np.square(df['Weight gained (grams)']), dist = "norm", plot = pylab)

#using reciprocal tranformation
stats.probplot(np.reciprocal(df['Weight gained (grams)']), dist = "norm", plot = pylab)


#using yeo johnson tranformation
tf = transformation.YeoJohnsonTransformer(variables = 'Weight gained (grams)')



df_tf = tf.fit_transform(df)

stats.probplot(df_tf['Weight gained (grams)'], dist="norm", plot=pylab)

'''
Data is now nearly normal
'''



