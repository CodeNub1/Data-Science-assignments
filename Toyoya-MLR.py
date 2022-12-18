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
df=pd.read_csv('C:/Users/Sidha/OneDrive/Desktop/Datasets/MLR/ToyotaCorolla.csv',encoding='unicode_escape')



df.info()
df.columns

#mssing values
df.isna().sum()


#duplicates
df.duplicated().sum()



#model building
import statsmodels.formula.api as smf

model = smf.ols('Price~Age_08_04+KM+HP+cc+Doors+Gears+Quarterly_Tax+Weight',data=df).fit()


#model testing
model.params

model.rsquared , model.rsquared_adj

#insignificant variable doors
#building SLR and MLR

slr=smf.ols('Price~Doors',data=df).fit()
slr.tvalues 
slr.pvalues



#variance inflation factor
rsq_Age_08_04=smf.ols('Age_08_04~KM+HP+cc+Doors+Gears+Quarterly_Tax+Weight',data=df).fit().rsquared
vif_Age_08_04=1/(1-rsq_Age_08_04)

rsq_KM=smf.ols('KM~Age_08_04+HP+cc+Doors+Gears+Quarterly_Tax+Weight',data=df).fit().rsquared
vif_KM=1/(1-rsq_KM)

rsq_HP=smf.ols('HP~Age_08_04+KM+cc+Doors+Gears+Quarterly_Tax+Weight',data=df).fit().rsquared
vif_HP=1/(1-rsq_HP)

rsq_CC=smf.ols('cc~Age_08_04+KM+HP+Doors+Gears+Quarterly_Tax+Weight',data=df).fit().rsquared
vif_CC=1/(1-rsq_CC)

rsq_DR=smf.ols('Doors~Age_08_04+KM+HP+cc+Gears+Quarterly_Tax+Weight',data=df).fit().rsquared
vif_DR=1/(1-rsq_DR)

rsq_GR=smf.ols('Gears~Age_08_04+KM+HP+cc+Doors+Quarterly_Tax+Weight',data=df).fit().rsquared
vif_GR=1/(1-rsq_GR)

rsq_QT=smf.ols('Quarterly_Tax~Age_08_04+KM+HP+cc+Doors+Gears+Weight',data=df).fit().rsquared
vif_QT=1/(1-rsq_QT)

rsq_WT=smf.ols('Weight~Age_08_04+KM+HP+cc+Doors+Gears+Quarterly_Tax',data=df).fit().rsquared
vif_WT=1/(1-rsq_WT)


# Storing vif values in a data frame
d1 = {'Variables':['Age_08_04','KM','HP','cc','Doors','Gears','Quarterly_Tax','Weight'],'VIF':[vif_Age_08_04,vif_KM,vif_HP,vif_CC,vif_DR,vif_GR,vif_QT,vif_WT]}
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




k = df.shape[1]
n = df.shape[0]
leverage_cutoff = 3*((k + 1)/n)

#high influence points
from statsmodels.graphics.regressionplots import influence_plot
fig,ax=plt.subplots(figsize=(15,10))
fig=influence_plot(model,ax = ax)

#improving model

df=df.drop(df.index[[221,960,80]],axis=0).reset_index(drop=True)



#final model
Final_model = smf.ols('Price~Age_08_04+KM+HP+cc+Doors+Gears+Quarterly_Tax+Weight',data=df).fit()


Final_model.summary()




#prediction
pred_y = Final_model.predict(df)










