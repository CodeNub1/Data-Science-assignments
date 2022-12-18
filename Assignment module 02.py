# -*- coding: utf-8 -*-
"""
Created on Sun Oct 23 08:16:09 2022

@author: Sidha
"""
from sympy import symbols, Eq
#relation for numbers for 399,543 and 12345
x=symbols('x')
a=12345
b=543
a/b
a%b
#therefore the equation will be
z=Eq(543*x+399-12345)
#Division and Modulus operator

a=-5/3
print(int(a))
b=-5%3
print(int(b))

# output of the following
a=5
b=3
c=10
# a=a/b
a/=b
print(a,b,c)
#c=c*5
c*=5
print(c)


's' in 'Data Science'
#exponential
4**3
