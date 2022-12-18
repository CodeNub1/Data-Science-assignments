# -*- coding: utf-8 -*-
"""
Created on Fri Dec 16 07:30:23 2022

@author: Sidha
"""

import lifelines
import pandas as pd
from lifelines import KaplanMeierFitter
import matplotlib.pyplot as plt
import seaborn as sns


#importing dataset
df=pd.read_excel('C:/Users/Sidha/OneDrive/Desktop/Datasets/Survival/ECG_Surv.xlsx')


#timeline
T=df.survival_time_hr
#event
E=df.alive 



# Initiating the KaplanMeierFitter model
kmf = KaplanMeierFitter()


kmf.fit(T,E,label='Kaplan Meier Estimate')


kmf.plot()
plt.show()





ax = plt.subplot(111)


kmf = KaplanMeierFitter()

for group in df['group'].unique():
    
    flag = df['group'] == group
    
    kmf.fit(T[flag], event_observed = E[flag], label = group)
    kmf.plot(ax=ax)

plt.title("Survival curves by Patient groups");







