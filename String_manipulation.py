# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 18:47:49 2022

@author: Sidha
"""

#creating string
a='Grow Gratitude'

#accessing letter G in growth
a[0]
#length of string
len(a)
#number of time letter G in the string
a.count('G')




b='Being awre of a single shortcoming within yourself is far more useful than being aware of a thousand in someone else'

#count number of characters
from collections import Counter

c=Counter(b)
print(c)





a='idealistic as it may sound, aktruism should be the driving force in business, not just compettition and a desire for wealth'
#get one char of the word
a[1]
#first three characters
a[0:3]
#last three characters
a[-3:]







a='stay positive and optimistic'
#split white spaces
a.split()
#starts with H
a.startswith('H')
#ends with d
a.endswith('d')
#ends with c
a.endswith('c')





a='o8'
#printing 108 time
b=a*108
print(b)

a='o'
#printing 108 time
b=a*108
print(b)









a='Grow Gratitude'
#replacing a string
a=a.replace('Grow', 'Growth of')
print(a)











a='.elgnuj eht otni ffo deps meht fo htoB .eerf noil eht tes ot sepor eht no dewangdnanar eh ,ylkciuQ .elbuortninoilehtdecitondnatsapdeklawesuomeht ,nooS .repmihwotdetratsdnatuotegotgnilggurts saw noilehT .eert a tsniagapumihdeityehT .mehthtiwnoilehtkootdnatserofehtotniemacsretnuhwef a ,yad enO .ogmihteldnaecnedifnocs’esuomeht ta dehgualnoilehT ”.emevasuoy fi yademosuoyotplehtaergfo eb lliw I ,uoyesimorp I“ .eerfmihtesotnoilehtdetseuqeryletarepsedesuomehtnehwesuomehttaeottuoba saw eH .yrgnaetiuqpuekow eh dna ,peels s’noilehtdebrutsidsihT .nufroftsujydobsihnwoddnapugninnurdetratsesuom a nehwelgnujehtnignipeelsecno saw noil A'
#reversing the string
a=a[::-1]
print(a)



