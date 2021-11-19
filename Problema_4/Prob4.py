# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 15:50:14 2021

@author: Andresda
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import random as rn
import math as math
import scipy as sp

x=[6,9,15]
y=[3,9,11]
yerr=[1,2,3]

a0=1
a1=1

def f(x,m,b):
    f=m*x+b
    return f

popt=sp.curve_fit(f,x,p0=[a0,a1])[0]
print(popt)

