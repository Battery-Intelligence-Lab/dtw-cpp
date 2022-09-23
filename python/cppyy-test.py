# -*- coding: utf-8 -*-
"""
Created on Fri Sep 23 15:37:17 2022

@author: laengs2321
"""

from cppyy.gbl import std
import cppyy

cppyy.include('../dtwc/utility.hpp') 


from cppyy.gbl import dtwc


# a = dtwc.VecMatrix[float](3,3,-1);

# a[0,0] = 3

# print(a[0,0])
# print(a[2,2])

x_vec = std.vector[float](10);
y_vec = std.vector[float](12);

b = dtwc.dtwFun2[float](x_vec, y_vec)






#load_data[float,True]('../data/dummy',3)


x = std.vector['int'](5)

print(x)