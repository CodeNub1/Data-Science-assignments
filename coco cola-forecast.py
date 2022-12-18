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










import pandas as pd
import numpy as np
from numpy import sqrt
import matplotlib.pyplot as plt
import seaborn as sns


#importing data set
df=pd.read_csv('C:/Users/Sidha/OneDrive/Desktop/Datasets/Forecasting/PlasticSales.csv')


df.info()


#duplicates
df.duplicated().sum()

df.isna().sum()


#Clean way to convert quarterly periods to datetime in pandas
df['Month_Year'] = df['Month'].str.split('-').apply(lambda x: ' 19'.join(x[:]))



#dropping prevous columns
df.drop(columns=['Month'],inplace=True)


## converted date column to index
df.set_index('Month_Year',inplace=True)

#checking plots
df.plot()
plt.show()





df.hist()
plt.show()






#moving average
plt.figure(figsize=(12,4))
df.Sales.plot(label="org")
for i in range(2,24,6):
    df["Sales"].rolling(i).mean().plot(label=str(i))
plt.legend(loc='best')




#Time series decomposition plot
from statsmodels.tsa.seasonal import seasonal_decompose
decompose_ts_add = seasonal_decompose(df.Sales,freq=12)  
decompose_ts_add.plot()
plt.show()





#ACF plots and PACF plots
import statsmodels.graphics.tsaplots as tsa_plots

tsa_plots.plot_acf(df.Sales,lags=14)
tsa_plots.plot_pacf(df.Sales,lags=14)
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
ses_model = SimpleExpSmoothing(Train["Sales"]).fit(smoothing_level=0.2)
pred_ses = ses_model.predict(start = Test.index[0],end = Test.index[-1])
MAPE(pred_ses,Test.Sales)


# Holt method 
from statsmodels.tsa.holtwinters import Holt # Holts Exponential Smoothing
hw_model = Holt(Train["Sales"]).fit(smoothing_level=0.1, smoothing_slope=0.2)
pred_hw = hw_model.predict(start = Test.index[0],end = Test.index[-1])
MAPE(pred_hw,Test.Sales)



#Holts winter
from statsmodels.tsa.holtwinters import ExponentialSmoothing
hwe_model_add_add = ExponentialSmoothing(Train["Sales"],seasonal="add",trend="add",seasonal_periods=12).fit(smoothing_level=0.1, smoothing_slope=0.2) #add the trend to the model
pred_hwe_add_add = hwe_model_add_add.predict(start = Test.index[0],end = Test.index[-1])
MAPE(pred_hwe_add_add,Test.Sales)


#Holts winter with multiplicative seasonality
hwe_model_mul_add = ExponentialSmoothing(Train["Sales"],seasonal="mul",trend="add",seasonal_periods=12).fit(smoothing_level=0.1, smoothing_slope=0.2) 
pred_hwe_mul_add = hwe_model_mul_add.predict(start = Test.index[0],end = Test.index[-1])
MAPE(pred_hwe_mul_add,Test.Sales)


from sklearn.metrics import mean_squared_error
rmse_hwe_mul_add = sqrt(mean_squared_error(pred_hwe_mul_add,Test.Sales))
rmse_hwe_mul_add


#Final Model
hwe_model_add_add = ExponentialSmoothing(df["Sales"],seasonal="add",trend="add",seasonal_periods=10).fit()


#Forecasting for next 10 time periods
hwe_model_add_add.forecast(10)







    

import pandas as pd
import numpy as np
from numpy import sqrt
import matplotlib.pyplot as plt
import seaborn as sns


#importing data set
df=pd.read_excel('C:/Users/Sidha/OneDrive/Desktop/Datasets/Forecasting/Airlines Data.xlsx')


df.info()


#duplicates
df.duplicated().sum()

df.isna().sum()


df.set_index('Month',inplace=True)
## making the month column as index


#checking plots
df.plot()
plt.show()



df.hist()
plt.show()

#moving average
plt.figure(figsize=(12,4))
df.Passengers.plot(label="org")
for i in range(2,24,6):
    df["Passengers"].rolling(i).mean().plot(label=str(i))
plt.legend(loc='best')

#Time series decomposition plot
from statsmodels.tsa.seasonal import seasonal_decompose
decompose_ts_add = seasonal_decompose(df.Passengers,freq=12)  
decompose_ts_add.plot()
plt.show()


#ACF plots and PACF plots
import statsmodels.graphics.tsaplots as tsa_plots

tsa_plots.plot_acf(df.Passengers,lags=14)
tsa_plots.plot_pacf(df.Passengers,lags=14)
plt.show()

#Evaluation Metric MAPE
def MAPE(pred,org):
    temp = np.abs((pred-org)/org)*100
    return np.mean(temp)




#splitting data into train and test
Train=df.head(81)
Test=df.tail(15)


#Simple Exponential Method
from statsmodels.tsa.holtwinters import SimpleExpSmoothing # SES
ses_model = SimpleExpSmoothing(Train["Passengers"]).fit(smoothing_level=0.2)
pred_ses = ses_model.predict(start = Test.index[0],end = Test.index[-1])
MAPE(pred_ses,Test.Passengers)


# Holt method 
from statsmodels.tsa.holtwinters import Holt # Holts Exponential Smoothing
hw_model = Holt(Train["Passengers"]).fit(smoothing_level=0.1, smoothing_slope=0.2)
pred_hw = hw_model.predict(start = Test.index[0],end = Test.index[-1])
MAPE(pred_hw,Test.Passengers)



#Holts winter
from statsmodels.tsa.holtwinters import ExponentialSmoothing
hwe_model_add_add = ExponentialSmoothing(Train["Passengers"],seasonal="add",trend="add",seasonal_periods=12).fit(smoothing_level=0.1, smoothing_slope=0.2) #add the trend to the model
pred_hwe_add_add = hwe_model_add_add.predict(start = Test.index[0],end = Test.index[-1])
MAPE(pred_hwe_add_add,Test.Passengers)


#Holts winter with multiplicative seasonality
hwe_model_mul_add = ExponentialSmoothing(Train["Passengers"],seasonal="mul",trend="add",seasonal_periods=12).fit(smoothing_level=0.1, smoothing_slope=0.2) 
pred_hwe_mul_add = hwe_model_mul_add.predict(start = Test.index[0],end = Test.index[-1])
MAPE(pred_hwe_mul_add,Test.Passengers)


from sklearn.metrics import mean_squared_error
rmse_hwe_mul_add = sqrt(mean_squared_error(pred_hwe_mul_add,Test.Passengers))
rmse_hwe_mul_add


#Final Model
hwe_model_add_add = ExponentialSmoothing(df["Passengers"],seasonal="add",trend="add",seasonal_periods=10).fit()


#Forecasting for next 10 time periods
hwe_model_add_add.forecast(10)




















import pandas as pd
import numpy as np
from numpy import sqrt
import matplotlib.pyplot as plt
import seaborn as sns


#importing data set
df=pd.read_excel('C:/Users/Sidha/OneDrive/Desktop/Datasets/Forecasting/CocaCola_Sales_Rawdata.xlsx')


df.info()


#duplicates
df.duplicated().sum()

df.isna().sum()


#Clean way to convert quarterly periods to datetime in pandas
df['Quarter_Year'] = df['Quarter'].str.split('_').apply(lambda x: ' 19'.join(x[:]))

# converting into datetime formate as the index was not in correct formate.
df['date'] = (
    pd.to_datetime(
        df['Quarter_Year'].str.split(' ').apply(lambda x: ''.join(x[::-1]))
,dayfirst=True))



#dropping prevous columns
df.drop(columns=['Quarter','Quarter_Year'],inplace=True)


## converted date column to index
df.set_index('date',inplace=True)

#checking plots
df.plot()
plt.show()





df.hist()
plt.show()






#moving average
plt.figure(figsize=(12,4))
df.Sales.plot(label="org")
for i in range(2,24,6):
    df["Sales"].rolling(i).mean().plot(label=str(i))
plt.legend(loc='best')




#Time series decomposition plot
from statsmodels.tsa.seasonal import seasonal_decompose
decompose_ts_add = seasonal_decompose(df.Sales,freq=12)  
decompose_ts_add.plot()
plt.show()





#ACF plots and PACF plots
import statsmodels.graphics.tsaplots as tsa_plots

tsa_plots.plot_acf(df.Sales,lags=14)
tsa_plots.plot_pacf(df.Sales,lags=14)
plt.show()




#Evaluation Metric MAPE
def MAPE(pred,org):
    temp = np.abs((pred-org)/org)*100
    return np.mean(temp)




#splitting data into train and test
Train=df.head(27)
Test=df.tail(15)


#Simple Exponential Method
from statsmodels.tsa.holtwinters import SimpleExpSmoothing # SES
ses_model = SimpleExpSmoothing(Train["Sales"]).fit(smoothing_level=0.2)
pred_ses = ses_model.predict(start = Test.index[0],end = Test.index[-1])
MAPE(pred_ses,Test.Sales)


# Holt method 
from statsmodels.tsa.holtwinters import Holt # Holts Exponential Smoothing
hw_model = Holt(Train["Sales"]).fit(smoothing_level=0.1, smoothing_slope=0.2)
pred_hw = hw_model.predict(start = Test.index[0],end = Test.index[-1])
MAPE(pred_hw,Test.Sales)



#Holts winter
from statsmodels.tsa.holtwinters import ExponentialSmoothing
hwe_model_add_add = ExponentialSmoothing(Train["Sales"],seasonal="add",trend="add",seasonal_periods=12).fit(smoothing_level=0.1, smoothing_slope=0.2) #add the trend to the model
pred_hwe_add_add = hwe_model_add_add.predict(start = Test.index[0],end = Test.index[-1])
MAPE(pred_hwe_add_add,Test.Sales)


#Holts winter with multiplicative seasonality
hwe_model_mul_add = ExponentialSmoothing(Train["Sales"],seasonal="mul",trend="add",seasonal_periods=12).fit(smoothing_level=0.1, smoothing_slope=0.2) 
pred_hwe_mul_add = hwe_model_mul_add.predict(start = Test.index[0],end = Test.index[-1])
MAPE(pred_hwe_mul_add,Test.Sales)


from sklearn.metrics import mean_squared_error
rmse_hwe_mul_add = sqrt(mean_squared_error(pred_hwe_mul_add,Test.Sales))
rmse_hwe_mul_add


#Final Model
hwe_model_add_add = ExponentialSmoothing(df["Sales"],seasonal="add",trend="add",seasonal_periods=10).fit()


#Forecasting for next 10 time periods
hwe_model_add_add.forecast(10)










