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
df=pd.read_csv("C:/Users/Sidha/OneDrive/Desktop/Datasets/Simple linear regression/Salary_Data.csv")
df.columns


#AUTO EDA
import sweetviz as sv


#analyze dataset
report=sv.analyze(df)

#display report
report.show_notebook()
report.show_html('EDAreport.html')


#correlation
df.corr()

sns.regplot(x=df['YearsExperience'],y=df['Salary'])

#import linear regression library
import statsmodels.formula.api as smf


#simple linear regression
model=smf.ols('Salary~YearsExperience',data=df).fit()
model.summary()


#coefficients
model.params




#model building
pred1 = model.predict(pd.DataFrame(df['YearsExperience']))


#regression line
plt.scatter(df.YearsExperience, df.Salary)
plt.plot(df.YearsExperience, pred1, "r")
plt.legend(['Observed data', 'Predicted line'])
plt.show()

#error calculation

res1=df.YearsExperience-pred1
res1

print(np.mean(res1))

res_sqr1 = res1 * res1
mse1 = np.mean(res_sqr1)
rmse1 = np.sqrt(mse1)
rmse1




#transormation using log
plt.scatter(x = np.log(df['YearsExperience']), y = df['Salary'], color = 'brown')
np.corrcoef(np.log(df.YearsExperience),df.Salary)

model2=smf.ols('Salary~np.log(YearsExperience)',data=df).fit()
model2.summary()

pred2 = model.predict(pd.DataFrame(df['YearsExperience']))

plt.scatter(np.log(df.YearsExperience), df.Salary)
plt.plot(np.log(df.YearsExperience), pred1, "r")
plt.legend(['Observed data', 'Predicted line'])
plt.show()


res2=df.YearsExperience-pred2
res2

print(np.mean(res2))

res_sqr2 = res2 * res2
mse2 = np.mean(res_sqr2)
rmse2 = np.sqrt(mse2)
rmse2



#transormation using exponential 
plt.scatter(x = df['YearsExperience'], y = np.log(df['Salary']), color = 'brown')
np.corrcoef(df.YearsExperience,np.log(df.Salary),)

model3=smf.ols('np.log(Salary)~YearsExperience',data=df).fit()
model3.summary()

pred3 = model.predict(pd.DataFrame(df['YearsExperience']))

plt.scatter(df.YearsExperience, np.log(df.Salary))
plt.plot(df.YearsExperience, pred3, "r")
plt.legend(['Observed data', 'Predicted line'])
plt.show()


res3=df.YearsExperience-pred3
res3

print(np.mean(res3))

res_sqr3 = res3 * res3
mse3 = np.mean(res_sqr3)
rmse3 = np.sqrt(mse3)
rmse3



#transormation using polynomial 

model4=smf.ols('np.log(Salary)~YearsExperience+I(YearsExperience*YearsExperience)',data=df).fit()
model4.summary()

pred4 = model.predict(pd.DataFrame(df['YearsExperience']))

#regression line
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2)
X = df.iloc[:, 0:1].values
X_poly = poly_reg.fit_transform(X)
print(X_poly)


plt.scatter(df.YearsExperience, np.log(df.Salary))
plt.plot(X, pred4, color = 'red')
plt.legend(['Predicted line', 'Observed data'])
plt.show()


#error calculation
res4=df.YearsExperience-pred4
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


