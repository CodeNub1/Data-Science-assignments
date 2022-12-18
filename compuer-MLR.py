# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 06:05:03 2022

@author: Sidha
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


#importing dataset
df=pd.read_csv('C:/Users/Sidha/OneDrive/Desktop/Datasets/MLR/Computer_Data.csv')



df.info()
df.columns

#dropping unwanted columns
df=df.drop('Unnamed: 0',axis=1)

#mssing values
df.isna().sum()


#duplicates
df.duplicated().sum()

df=df.drop_duplicates()


#creating dummy variables
df=pd.get_dummies(df,columns=['cd','multi','premium'],drop_first=True)

#model building
import statsmodels.formula.api as smf

model = smf.ols('price~speed+hd+ram+screen+ads+trend+cd_yes+multi_yes+premium_yes',data=df).fit()


#model testing
model.params

model.rsquared , model.rsquared_adj


model.tvalues
model.pvalues


model.summary()




#variance inflation factor
rsq_price = smf.ols('price~speed+hd+ram+screen+ads+trend+cd_yes+multi_yes+premium_yes',data=df).fit().rsquared  
vif_price = 1/(1-rsq_price) 

rsq_speed = smf.ols('speed~price+hd+ram+screen+ads+trend+cd_yes+multi_yes+premium_yes',data=df).fit().rsquared  
vif_speed = 1/(1-rsq_speed) 

rsq_hd = smf.ols('hd~speed+price+ram+screen+ads+trend+cd_yes+multi_yes+premium_yes',data=df).fit().rsquared  
vif_hd = 1/(1-rsq_hd) 

rsq_ram = smf.ols('ram~speed+hd+price+screen+ads+trend+cd_yes+multi_yes+premium_yes',data=df).fit().rsquared  
vif_ram = 1/(1-rsq_ram) 

rsq_screen = smf.ols('screen~speed+hd+ram+price+ads+trend+cd_yes+multi_yes+premium_yes',data=df).fit().rsquared  
vif_screen = 1/(1-rsq_screen) 

rsq_ads = smf.ols('ads~speed+hd+ram+screen+price+trend+cd_yes+multi_yes+premium_yes',data=df).fit().rsquared  
vif_ads = 1/(1-rsq_ads) 

rsq_trend = smf.ols('trend~speed+hd+ram+screen+ads+price+cd_yes+multi_yes+premium_yes',data=df).fit().rsquared  
vif_trend = 1/(1-rsq_trend) 

rsq_cd = smf.ols('cd_yes~speed+hd+ram+screen+ads+trend+price+multi_yes+premium_yes',data=df).fit().rsquared  
vif_cd = 1/(1-rsq_cd) 

rsq_multi = smf.ols('multi_yes~speed+hd+ram+screen+ads+trend+cd_yes+price+premium_yes',data=df).fit().rsquared  
vif_multi = 1/(1-rsq_multi) 

rsq_prm = smf.ols('premium_yes~speed+hd+ram+screen+ads+trend+cd_yes+multi_yes+price',data=df).fit().rsquared  
vif_prm = 1/(1-rsq_prm) 


# Storing vif values in a data frame
d1 = {'Variables':['price', 'speed', 'hd', 'ram', 'screen', 'ads', 'trend', 'cd_yes','multi_yes', 'premium_yes'],'VIF':[vif_price,vif_speed,vif_hd,vif_ram,vif_screen,vif_ads,vif_trend,vif_cd,vif_multi,vif_prm]}
Vif_frame = pd.DataFrame(d1)  
Vif_frame




#detecting influencers and outliers
model_influence = model.get_influence()
(c, _) = model_influence.cooks_distance


#Plot the influencers values using stem plot
sns.set_style(style='darkgrid')
fig = plt.subplots(figsize=(20, 7))
plt.stem(np.arange(len(df)), np.round(c, 3))
plt.xlabel('Row index')
plt.ylabel('Cooks Distance')
plt.show()


#high influence points
from statsmodels.graphics.regressionplots import influence_plot
fig,ax=plt.subplots(figsize=(15,10))
fig=influence_plot(model,ax = ax)




k = df.shape[1]
n = df.shape[0]
leverage_cutoff = 3*((k + 1)/n)
leverage_cutoff
#improving model

df=df.drop(df.index[[900,1101,1610,1524,2042,1835,720,1048,1688,1784,5960,4477,3783,2281]],axis=0).reset_index()



#final model
Final_model = smf.ols('price~speed+hd+ram+screen+ads+trend+cd_yes+multi_yes+premium_yes',data=df).fit()


Final_model.summary()




#prediction
pred_y = Final_model.predict(df)










