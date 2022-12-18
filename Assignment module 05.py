# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 11:28:58 2022

@author: Sidha
"""
#all fiucntion and parameters
list1=[1,5.5,10+2j,'data science']
all_functions=(dir(list1))
print(all_functions)

#create a sequence of numbers
a=list(range(3,7))

#take input to create list
x=int(input(" Enter the lower limit:\n"))
y=int(input('Enter the upper limit:'))

print(list(range(x,y+1)))

#creating dictionaries from  2 lists
a=list(range(0,10))
print(a)
b = list(map(str, input("Enter multiple values: ").split())) 
print(b)
len(a)
di={0:'zero'}
for i in range(len(a)):
    di[b[i]]=a[i]


print(di)

#3
def odd_check(number):
    if number%2==0:
        return('False')
    else:
        return('True')

list1=list(range(3,9))
print(list1)


for i in range(len(list1)):
    if odd_check(list1[i])=='True':
        list1[i]+=10
    else:
       list1[i]*=5 
       
       
       
print(list1)  


     
#4 UDF

def m(name,message):
    name=str(input("Enter your name:"))
    message=str(input("Enter your message"))
    if message=='':
        message=("Have a great day")
        print('Hello %s %s'%(name,message))
    else:
        print('Hello %s %s'%(name,message))
   
  

print(m(a,b))
