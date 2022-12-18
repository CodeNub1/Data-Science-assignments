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
df=pd.read_csv("C:/Users/Sidha/OneDrive/Desktop/Datasets/Simple linear regression/calories_consumed.csv")
df.columns
df=df.rename({'Weight gained (grams)':'weight_gained','Calories Consumed':'calories_consumed'},axis=1)


#AUTO EDA
import sweetviz as sv


#analyze dataset
report=sv.analyze(df)

#display report
report.show_notebook()
report.show_html('EDAreport.html')


#correlation
df.corr()

sns.regplot(x=df['weight_gained'],y=df['calories_consumed'])

#import linear regression library
import statsmodels.formula.api as smf


#simple linear regression
model=smf.ols('calories_consumed~weight_gained',data=df).fit()
model.summary()


#coefficients
model.params




#model building
pred1 = model.predict(pd.DataFrame(df['weight_gained']))


#regression line
plt.scatter(df.weight_gained, df.calories_consumed)
plt.plot(df.weight_gained, pred1, "r")
plt.legend(['Observed data', 'Predicted line'])
plt.show()

#error calculation
res1=df.weight_gained-pred1
res1

print(np.mean(res1))

res_sqr1 = res1 * res1
mse1 = np.mean(res_sqr1)
rmse1 = np.sqrt(mse1)
rmse1




#transormation using log
plt.scatter(x = np.log(df['weight_gained']), y = df['calories_consumed'], color = 'brown')
np.corrcoef(np.log(df.weight_gained),df.calories_consumed)

model2=smf.ols('calories_consumed~np.log(weight_gained)',data=df).fit()
model2.summary()

pred2 = model.predict(pd.DataFrame(df['weight_gained']))

plt.scatter(np.log(df.weight_gained), df.calories_consumed)
plt.plot(np.log(df.weight_gained), pred1, "r")
plt.legend(['Observed data', 'Predicted line'])
plt.show()


res2=df.weight_gained-pred2
res2

print(np.mean(res2))

res_sqr2 = res2 * res2
mse2 = np.mean(res_sqr2)
rmse2 = np.sqrt(mse2)
rmse2



#transormation using exponential 
plt.scatter(x = df['weight_gained'], y = np.log(df['calories_consumed']), color = 'brown')
np.corrcoef(df.weight_gained,np.log(df.calories_consumed),)

model3=smf.ols('np.log(calories_consumed)~weight_gained',data=df).fit()
model3.summary()

pred3 = model.predict(pd.DataFrame(df['weight_gained']))

plt.scatter(df.weight_gained, np.log(df.calories_consumed))
plt.plot(df.weight_gained, pred3, "r")
plt.legend(['Observed data', 'Predicted line'])
plt.show()


res3=df.weight_gained-pred3
res3

print(np.mean(res3))

res_sqr3 = res3 * res3
mse3 = np.mean(res_sqr3)
rmse3 = np.sqrt(mse3)
rmse3



#transormation using polynomial 

model4=smf.ols('np.log(calories_consumed)~weight_gained+I(weight_gained*weight_gained)',data=df).fit()
model4.summary()

pred4 = model.predict(pd.DataFrame(df['weight_gained']))

#regression line
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2)
X = df.iloc[:, 0:1].values
X_poly = poly_reg.fit_transform(X)
print(X_poly)


plt.scatter(df.weight_gained, np.log(df.calories_consumed))
plt.plot(X, pred4, color = 'red')
plt.legend(['Predicted line', 'Observed data'])
plt.show()


#error calculation
pred4_at=np.exp(pred4)


res4=df.weight_gained-pred4
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


