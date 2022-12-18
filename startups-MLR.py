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
df=pd.read_csv('C:/Users/Sidha/OneDrive/Desktop/Datasets/MLR/50_Startups.csv')



df.info()
df.columns


#renaming columns
df.rename(columns = {'R&D Spend':'RD', 'Marketing Spend':'MS'}, inplace = True)

#mssing values
df.isna().sum()


#duplicates
df.duplicated().sum()



#model building
import statsmodels.formula.api as smf

model = smf.ols('Profit~RD+Administration+MS',data=df).fit()


#model testing
model.params

model.rsquared , model.rsquared_adj


model.tvalues
model.pvalues


model.summary()




#variance inflation factor
rsq_pro = smf.ols('Profit~RD+Administration+MS',data=df).fit().rsquared  
vif_pro = 1/(1-rsq_pro) 

rsq_rd = smf.ols('RD~Profit+Administration+MS',data=df).fit().rsquared  
vif_rd = 1/(1-rsq_rd) 

rsq_adm = smf.ols('Administration~Profit+RD+MS',data=df).fit().rsquared  
vif_adm = 1/(1-rsq_adm)

rsq_ms = smf.ols('MS~Administration+Profit+RD',data=df).fit().rsquared  
vif_ms = 1/(1-rsq_ms)


# Storing vif values in a data frame
d1 = {'Variables':['pro','rd','adm','ms'],'VIF':[vif_pro,vif_rd,vif_adm,vif_ms]}
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

#improving model

df=df.drop(df.index[[49]],axis=0).reset_index()



#final model
Final_model = smf.ols('Profit~RD+Administration+MS',data=df).fit()


Final_model.summary()




#prediction
pred_y = Final_model.predict(df)










