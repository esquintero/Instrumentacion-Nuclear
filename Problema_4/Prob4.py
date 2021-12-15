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
import scipy.optimize as sp

x=[6,9,15]
y=[11,9,3]
yerr=[3,2,1]

a0=1
a1=1

def fun(x,m,b):
    f=m*x+b
    return f
#curva=fun(x,-0.74,16.72)
popt=sp.curve_fit(fun,x,y,p0=[a0,a1])[0]
print(popt)
sigmax=np.std(x)
a=np.sqrt((3.75/6)**2+(1/9)**2)
yee=np.zeros(len(x))

for i in range(len(x)):
    
    yee[i]=fun(x[i],-0.904,16.714)
    

plt.errorbar(x,y,yerr,marker='s',ls="",label="Dato e incertidumbre")
plt.plot(x,yee,label='Ajuste')
plt.legend(loc=0)