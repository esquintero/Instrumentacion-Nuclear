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
#Valores que se usan para el fitting (sugerencia al programa)
m,b=1,160
#Vector de # de medida y medicion
d=[147,152,153,171,146,168,145,133,168,171]
x=np.arange(1,11)
#Vector de incertidumbre (usando raíz cuadrada) redondeado a dos cifras
yerror=np.sqrt(d)
for i in range(len(yerror)):
    yerror[i]=round(yerror[i],2)

print(yerror)

#Sigma y la media de los datos
sigma=np.std(d)
mu=np.mean(d)
print(sigma,mu)

#Recta para graficar el fitting
def f(x,m,b):
    f=m*x+b
    return f
popt,popc=sp.curve_fit(f,x,d,p0=[m,b])

#Incertidumbre de el valor medio

def incertmedia(yerror):
    mu_error=0
    for i in range(len(yerror)):
        mu_error+=yerror[i]**2
    mu_error=np.sqrt(mu_error)
    return mu_error

mu_error=incertmedia(yerror)
print(round(mu_error,2))
#plt.plot(x,d,"o")
plt.errorbar(x,d,yerror,marker='s',ls="",label="Medición")
#plt.plot(x,f(x,*popt))
plt.hlines(mu,xmin=0,xmax=10,color='k',label="Valor medio")
plt.legend(loc=2)
plt.xticks(x)