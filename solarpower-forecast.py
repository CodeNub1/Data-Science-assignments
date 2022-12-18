# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 10:41:33 2022

@author: Sidha
"""

import pandas as pd
import numpy as np
from numpy import sqrt
import matplotlib.pyplot as plt
import seaborn as sns


#importing data set
df=pd.read_csv('C:/Users/Sidha/OneDrive/Desktop/Datasets/Forecasting/solarpower_cumuldaybyday2.csv')
df.columns

df.info()


#duplicates
df.duplicated().sum()

df.isna().sum()


## converted date column to index
df.set_index('date',inplace=True)

#checking plots
df.plot()
plt.show()





df.hist()
plt.show()






#moving average
plt.figure(figsize=(12,4))
df.cum_power.plot(label="org")
for i in range(2,24,6):
    df["cum_power"].rolling(i).mean().plot(label=str(i))
plt.legend(loc='best')




#Time series decomposition plot
from statsmodels.tsa.seasonal import seasonal_decompose
decompose_ts_add = seasonal_decompose(df.cum_power,freq=12)  
decompose_ts_add.plot()
plt.show()





#ACF plots and PACF plots
import statsmodels.graphics.tsaplots as tsa_plots

tsa_plots.plot_acf(df.cum_power,lags=14)
tsa_plots.plot_pacf(df.cum_power,lags=14)
plt.show()




#Evaluation Metric MAPE
def MAPE(pred,org):
    temp = np.abs((pred-org)/org)*100
    return np.mean(temp)




#splitting data into train and test
Train=df.head(40)
Test=df.tail(20)


#Simple Exponential Method
from statsmodels.tsa.holtwinters import SimpleExpSmoothing # SES
ses_model = SimpleExpSmoothing(Train["cum_power"]).fit(smoothing_level=0.2)
pred_ses = ses_model.predict(start = Test.index[0],end = Test.index[-1])
MAPE(pred_ses,Test.cum_power)


# Holt method 
from statsmodels.tsa.holtwinters import Holt # Holts Exponential Smoothing
hw_model = Holt(Train["cum_power"]).fit(smoothing_level=0.1, smoothing_slope=0.2)
pred_hw = hw_model.predict(start = Test.index[0],end = Test.index[-1])
MAPE(pred_hw,Test.cum_power)



#Holts winter
from statsmodels.tsa.holtwinters import ExponentialSmoothing
hwe_model_add_add = ExponentialSmoothing(Train["cum_power"],seasonal="add",trend="add",seasonal_periods=12).fit(smoothing_level=0.1, smoothing_slope=0.2) #add the trend to the model
pred_hwe_add_add = hwe_model_add_add.predict(start = Test.index[0],end = Test.index[-1])
MAPE(pred_hwe_add_add,Test.cum_power)


#Holts winter with multiplicative seasonality
hwe_model_mul_add = ExponentialSmoothing(Train["cum_power"],seasonal="mul",trend="add",seasonal_periods=12).fit(smoothing_level=0.1, smoothing_slope=0.2) 
pred_hwe_mul_add = hwe_model_mul_add.predict(start = Test.index[0],end = Test.index[-1])
MAPE(pred_hwe_mul_add,Test.cum_power)


from sklearn.metrics import mean_squared_error
rmse_hwe_mul_add = sqrt(mean_squared_error(pred_hwe_mul_add,Test.cum_power))
rmse_hwe_mul_add


#Final Model
hwe_model_add_add = ExponentialSmoothing(df["cum_power"],seasonal="add",trend="add",seasonal_periods=10).fit()


#Forecasting for next 10 time periods
hwe_model_add_add.forecast(10)










