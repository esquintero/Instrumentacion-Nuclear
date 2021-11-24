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
    yerror[i]=np.sqrt(d[i])
    yerror[i]=round(yerror[i],2)

print(yerror)

#Sigma y la media de los datos
sigma=np.std(d)
mu=np.mean(d)
print(sigma,mu)

def desv(x):
    suma=0
    for i in range(len(x)):
        suma+=np.power(mu-x[i],2)
    sigma=np.sqrt(suma/10.)
    return sigma    

sigma2=desv(d)
sigma3=np.sqrt(mu)
print(sigma,sigma2,sigma3,mu)
"""
#Recta para graficar el fitting
def f(x,m,b):
    f=m*x+b
    return f
popt,popc=sp.curve_fit(f,x,d,p0=[m,b])
"""
#Incertidumbre usando el valor medio
def errorconmu(d,mu):
    y_error=np.zeros(len(d))
    for i in range(len(d)):
        y_error[i]=d[i]-mu
    return y_error

    
yerror2=errorconmu(d, mu)
print(yerror2)

#Incertidumbre de el valor medio


def incertmedia(yerror):
    mu_error=0
    for i in range(len(yerror)):
        mu_error+=np.power(yerror[i],2)/10.
    print(mu_error)    
    mu_error=np.sqrt(mu_error)
    return mu_error


mu_error=incertmedia(yerror)
print(round(mu_error,2))


#plt.plot(x,d,"o")
plt.errorbar(x,d,yerror,marker='s',ls="",label="Medición")
plt.title("Medición con detector Geiger-Müller")
plt.xlabel("Medida")
plt.ylabel("Cuentas")
#plt.plot(x,f(x,*popt))
plt.hlines([mu,mu+mu_error,mu-mu_error],xmin=0,xmax=10,color='k',label="Valor medio y desviación")
#plt.hlines(mu,xmin=0,xmax=10,color='k',label="Valor medio")

plt.legend(loc=2)
plt.xticks(x)
#plt.savefig("p2g3")