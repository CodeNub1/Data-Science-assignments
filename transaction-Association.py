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

transaction = []
with open(r"C:/Users/Sidha/OneDrive/Desktop/Datasets/Association/transactions_retail.csv") as f:
    transaction = f.read()

transaction



#splitting data usng separator
transaction=transaction.split("\n")
                                                                                                                                                                                                                                                                                                                                                                                                       

#putting numbers for each item to make a countable table
df1=pd.DataFrame(transaction)


#identifying each item
transaction_list = []
for i in transaction:
    transaction_list.append(i.split(","))
    
print(transaction_list)


#using Transaction coder
te=TransactionEncoder()
x=te.fit(transaction_list).transform(transaction_list)

te.columns_

df=pd.DataFrame(x,columns=te.columns_)

df.columns

df.drop(df.columns[[0,1]],axis=1,inplace=True)




#Association rules with 10% support and 70% confidence
df_a=apriori(df,min_support=0.01,use_colnames=True)

#most frequesnt itemsets based on support
df_a.sort_values('support',ascending=False,inplace=True)
df_a.info()



#association rules
rules=association_rules(df_a,metric='lift',min_threshold=0.7)

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
