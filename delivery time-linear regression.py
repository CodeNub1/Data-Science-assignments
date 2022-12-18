# -*- coding: utf-8 -*-
"""
Created on Sat Dec  3 17:01:26 2022

@author: Sidha
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



#import dataset
df=pd.read_csv("C:/Users/Sidha/OneDrive/Desktop/Datasets/Simple linear regression/delivery_time.csv")
df.columns
df=df.rename({'Delivery Time':'delivery_time','Sorting Time':'sorting_time'},axis=1)


#AUTO EDA
import sweetviz as sv


#analyze dataset
report=sv.analyze(df)

#display report
report.show_notebook()
report.show_html('EDAreport.html')


#correlation
df.corr()

sns.regplot(x=df['delivery_time'],y=df['sorting_time'])

#import linear regression library
import statsmodels.formula.api as smf


#simple linear regression
model=smf.ols('sorting_time~delivery_time',data=df).fit()
model.summary()


#coefficients
model.params




#model building
pred1 = model.predict(pd.DataFrame(df['delivery_time']))


#regression line
plt.scatter(df.delivery_time, df.sorting_time)
plt.plot(df.delivery_time, pred1, "r")
plt.legend(['Observed data', 'Predicted line'])
plt.show()

#error calculation
res1=df.delivery_time-pred1
res1

print(np.mean(res1))

res_sqr1 = res1 * res1
mse1 = np.mean(res_sqr1)
rmse1 = np.sqrt(mse1)
rmse1




#transormation using log
plt.scatter(x = np.log(df['delivery_time']), y = df['sorting_time'], color = 'brown')
np.corrcoef(np.log(df.delivery_time),df.sorting_time)

model2=smf.ols('sorting_time~np.log(delivery_time)',data=df).fit()
model2.summary()

pred2 = model.predict(pd.DataFrame(df['delivery_time']))

plt.scatter(np.log(df.delivery_time), df.sorting_time)
plt.plot(np.log(df.delivery_time), pred1, "r")
plt.legend(['Observed data', 'Predicted line'])
plt.show()


res2=df.delivery_time-pred2
res2

print(np.mean(res1))

res_sqr2 = res2 * res2
mse2 = np.mean(res_sqr2)
rmse2 = np.sqrt(mse2)
rmse2



#transormation using exponential 

model3=smf.ols('np.log(sorting_time)~delivery_time',data=df).fit()
model3.summary()

pred3 = model.predict(pd.DataFrame(df['delivery_time']))

plt.scatter(df.delivery_time, np.log(df.sorting_time))
plt.plot(df.delivery_time, pred3, "r")
plt.legend(['Observed data', 'Predicted line'])
plt.show()


res3=df.delivery_time-pred3
res3

print(np.mean(res3))

res_sqr3 = res3 * res3
mse3 = np.mean(res_sqr3)
rmse3 = np.sqrt(mse3)
rmse3



#transormation using polynomial 

model4=smf.ols('np.log(sorting_time)~delivery_time+I(delivery_time*delivery_time)',data=df).fit()
model4.summary()

pred4 = model.predict(pd.DataFrame(df['delivery_time']))

#regression line
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2)
X = df.iloc[:, 0:1].values
X_poly = poly_reg.fit_transform(X)
print(X_poly)


plt.scatter(df.delivery_time, np.log(df.sorting_time))
plt.plot(X, pred4, color = 'red')
plt.legend(['Predicted line', 'Observed data'])
plt.show()


#error calculation
pred4_at=np.exp(pred4)


res4=df.delivery_time-pred4_at
res4

print(np.mean(res4))

res_sqr4 = res4 * res4
mse4 = np.mean(res_sqr4)
rmse4 = np.sqrt(mse4)
rmse4



#models and their rms values
data = {"MODEL":pd.Series(["SLR", "Log model", "Exp model", "Poly model"]), "RMSE":pd.Series([rmse1, rmse2, rmse3, rmse4])}
table_rmse = pd.DataFrame(data)
table_rmse




