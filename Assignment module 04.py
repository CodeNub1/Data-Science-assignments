# -*- coding: utf-8 -*-
"""
Created on Sun Oct 23 21:40:41 2022

@author: Sidha
"""
#1
age=50
type(age)
if age<=10:
    print('Children')
elif age>=60:
    print('senior citizen')
elif age>10 and age<60:
    print('normal citizen')
  
#2
i=0
j=0
price=7000
passenger_sex=['male','female']
passenger_type=['normal citizen','senior citizen']
if passenger_sex[i]=='male' and passenger_type[j]=='normal citizen':
    print(price*0.70)
elif passenger_sex[i]=='female' and passenger_type[j]=='senior citizen':
    print(price*0.50)
elif passenger_sex[i]=='female' and passenger_type[j]=='normal citizen':
    print(price*0.50)
else:
    print(price)


#3
def positive_check(number):
    if number<0:
        print('given number is negative')
    else:
        print('positive number')
        
def divisibility_5(number):
    if number%5==0:
        print('number is divisible by 5')
    else:
        print('number is not divisible by 5')
        

positive_check(5)
divisibility_5(26)


