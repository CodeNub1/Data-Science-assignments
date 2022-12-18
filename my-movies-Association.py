# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 11:41:12 2022

@author: Sidha
"""
#importing required packages
import warnings
warnings.filterwarnings("ignore")


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

#importing Dataset
df1=pd.read_csv('C:/Users/Sidha/OneDrive/Desktop/Datasets/Association/my_movies.csv')
df1.info()

df=df1.iloc[:,5:]
df


#Association rules with 10% support and 70% confidence
df_a=apriori(df,min_support=0.1,use_colnames=True)

#most frequesnt itemsets based on support
df_a.sort_values('support',ascending=False,inplace=True)
df_a.info()



#association rules
rules=association_rules(df_a,metric='lift',min_threshold=0.7)

#lift ratio greater than 1 is good rule to select the transactions
rules[rules.lift>1]

#visualizing the obtained rule
plt.scatter(rules['support'],rules['confidence'])



#Association rules with 20% support and 60% confidence
df_a=apriori(df,min_support=0.2,use_colnames=True)

#most frequesnt itemsets based on support
df_a.sort_values('support',ascending=False,inplace=True)
df_a.info()



#association rules
rules=association_rules(df_a,metric='lift',min_threshold=0.6)

#lift ratio greater than 1 is good rule to select the transactions
rules[rules.lift>1]

#visualizing the obtained rule
plt.scatter(rules['support'],rules['confidence'])




#Association rules with 5% support and 80% confidence
df_a=apriori(df,min_support=0.05,use_colnames=True)

#most frequesnt itemsets based on support
df_a.sort_values('support',ascending=False,inplace=True)
df_a.info()



#association rules
rules=association_rules(df_a,metric='lift',min_threshold=0.8)

#lift ratio greater than 1 is good rule to select the transactions
rules[rules.lift>1]

#visualizing the obtained rule
plt.scatter(rules['support'],rules['confidence'])







#Association rules with 5% support and 90% confidence
df_a=apriori(df,min_support=0.05,use_colnames=True)

#most frequesnt itemsets based on support
df_a.sort_values('support',ascending=False,inplace=True)
df_a.info()



#association rules
rules=association_rules(df_a,metric='lift',min_threshold=0.9)

#lift ratio greater than 1 is good rule to select the transactions
rules[rules.lift>1]

#visualizing the obtained rule
plt.scatter(rules['support'],rules['confidence'])




#Handling Profusion of Rules (Duplication elimination)

def to_list(i):
    return (sorted(list(i)))



ma_x=rules.antecedents.apply(to_list)+rules.consequents.apply(to_list)
ma_x

rules_sets=list(ma_x)


unique_rules_sets=[list(m) for m in set(tuple(i) for i in rules_sets)]
unique_rules_sets
