# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 21:43:03 2022

@author: Sidha
"""

#2 list with available data types
a=[9,9.5,'sid',5+2j,'False']
b=[8,8.5,'dis',3+2j,'True']





#concatenating list
c=a+b
print(c)




#Frequency of each element
for i in c:
    print(c.count(i))
    
    
    
    
#List in reversed order
c.reverse()
print(c)



#2sets
d={1,2,3,4,5,6,7,8,9,10}
e={5,6,7,8,9,10,11,12,13,14,15}

print(d[1])

#common elements in both sets
g=list(d)
h=list(e)
print(g,h)
i=0
j=0
while i<len(g) :
    while j<len(h):
        if g[i]==h[j]:
            print(g[i],'is common')
            j=j+1
        else:
            i=i+1
            break
         
#remove  element 7 from both the sets
print(d)
d.remove(7)
print('updated set',d)
e.remove(7)
print('updated set',e)





#Create dictionaries
ab={'kerala':123,'karnataka':111,'tamil nadu':222,'maharashtra':333,'telangana':223}
ab.keys()
ab.values()
ab['hyderabad']=212
print(ab)
