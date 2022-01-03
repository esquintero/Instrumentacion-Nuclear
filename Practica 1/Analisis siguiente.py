# -*- coding: utf-8 -*-
"""
Created on Mon Oct 18 14:38:12 2021

@author: Andresda
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import (AutoMinorLocator, MultipleLocator,AutoLocator)
from scipy.optimize import curve_fit
from ProgramaScipy import  muco57,mu1co60,mu2co60,mu1cs137,mu2cs137,mu1na22,mu2na22


a0,a1=1.,1.
a0=np.float64(a0)
a1=np.float64(a1)
x=[mu1cs137+.0,muco57+.0,mu2cs137+.0,mu1co60+.0,mu2na22+.0,mu2co60+.0]
y=[32,122.06,661.66,1173.228,1274,1332.490]

#print(type(x[0]))
#print(x,y)
#print(type(a0))
def recta(x,a0,a1):
    return(a0+a1*x)
#Se hace el ajuste de recta para poder graficar Energía vs canal
popt, pcov = curve_fit(recta, x, y, p0=[a0,a1])
a0=popt[0]
a1=popt[1]
print(type(a1))
x1=np.arange(0,1750,1)
print(popt)
fig=plt.figure(figsize=(8.,8.))
plt.plot(x,y,'o',label="Valores $\mu$")
plt.plot(x1, recta( x1,a0,a1),label="E=-9.3+0.75*Canal")
#plt.plot(x,recta(a0,a1,x))
plt.legend(loc = 2)
plt.title("Gráfica Ajuste Energía vs Canal")
plt.xlabel("Canal")
plt.ylabel("Energía(keV)")
plt.show()
fig.savefig("Grafica ajuste Energía vs Canal")
