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
df=pd.read_csv('C:/Users/Sidha/OneDrive/Desktop/Datasets/MLR/Avacado_Price.csv')



df.info()
df.columns


#renaming columns
df.rename(columns = {'XLarge Bags':'XLarge_Bags'}, inplace = True)


#mssing values
df.isna().sum()


#duplicates
df.duplicated().sum()



#model building
import statsmodels.formula.api as smf

model = smf.ols('AveragePrice~Total_Volume+tot_ava1+tot_ava2+tot_ava3+Total_Bags+Small_Bags+Large_Bags+XLarge_Bags',data=df).fit()


#model testing
model.params

model.rsquared , model.rsquared_adj


model.tvalues
model.pvalues


model.summary()




#variance inflation factor
rsq_price = smf.ols('AveragePrice~Total_Volume+tot_ava1+tot_ava2+tot_ava3+Total_Bags+Small_Bags+Large_Bags+XLarge_Bags',data=df).fit().rsquared  
vif_price = 1/(1-rsq_price) 

rsq_tv = smf.ols('Total_Volume~AveragePrice+tot_ava1+tot_ava2+tot_ava3+Total_Bags+Small_Bags+Large_Bags+XLarge_Bags',data=df).fit().rsquared  
vif_tv = 1/(1-rsq_tv) 

rsq_ta1 = smf.ols('tot_ava1~Total_Volume+AveragePrice+tot_ava2+tot_ava3+Total_Bags+Small_Bags+Large_Bags+XLarge_Bags',data=df).fit().rsquared  
vif_ta1 = 1/(1-rsq_ta1)

rsq_ta2 = smf.ols('tot_ava2~Total_Volume+tot_ava1+AveragePrice+tot_ava3+Total_Bags+Small_Bags+Large_Bags+XLarge_Bags',data=df).fit().rsquared  
vif_ta2 = 1/(1-rsq_ta2)


rsq_ta3 = smf.ols('tot_ava3~Total_Volume+tot_ava1+tot_ava2+AveragePrice+Total_Bags+Small_Bags+Large_Bags+XLarge_Bags',data=df).fit().rsquared  
vif_ta3 = 1/(1-rsq_ta3)

rsq_tb = smf.ols('Total_Bags~Total_Volume+tot_ava1+tot_ava2+tot_ava3+AveragePrice+Small_Bags+Large_Bags+XLarge_Bags',data=df).fit().rsquared  
vif_tb = 1/(1-rsq_tb)

rsq_sb = smf.ols('Small_Bags~Total_Volume+tot_ava1+tot_ava2+tot_ava3+Total_Bags+AveragePrice+Large_Bags+XLarge_Bags',data=df).fit().rsquared  
vif_sb = 1/(1-rsq_sb)

rsq_lb = smf.ols('Large_Bags~Total_Volume+tot_ava1+tot_ava2+tot_ava3+Total_Bags+Small_Bags+AveragePrice+XLarge_Bags',data=df).fit().rsquared  
vif_lb = 1/(1-rsq_lb)

rsq_xlb = smf.ols('XLarge_Bags~Total_Volume+tot_ava1+tot_ava2+tot_ava3+Total_Bags+Small_Bags+Large_Bags+AveragePrice',data=df).fit().rsquared  
vif_xlb = 1/(1-rsq_xlb)


# Storing vif values in a data frame
d1 = {'Variables':['AveragePrice','Total_Volume','tot_ava1','tot_ava2','tot_ava3','Total_Bags','Small_Bags','Large_Bags','XLarge_Bags'],'VIF':[vif_price,vif_tv,vif_ta1,vif_ta2,vif_ta3,vif_tb,vif_sb,vif_lb,vif_xlb]}
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

df=df.drop(df.index[[15560,17468]],axis=0).reset_index()



#final model
Final_model = smf.ols('AveragePrice~Total_Volume+tot_ava1+tot_ava2+tot_ava3+Total_Bags+Small_Bags+Large_Bags+XLarge_Bags',data=df).fit()


Final_model.summary()




#prediction
pred_y = Final_model.predict(df)










