# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 13:26:55 2021

@author: Andresda & Esteban & LoyaL
"""

import numpy as np
import matplotlib.pyplot as plt
e=np.power(1.602,-19)
auz=79.
alfaz=2.
proton=1.
omega=197.
I=0.000790
mec=0.511
mn=939.
malfa=4.*mn
mproton=1.*mn
ro=19.32
E=np.arange(1,1000000)
beta=np.sqrt(1-(1/np.power(1+(E/mn),2)))
gamma=1./np.sqrt(1-np.power(beta,2))

#Wmax=(2*mec*np.power(beta*gamma,2))/(1+2*(mec/malfa)*gamma+np.power(mec/mn,2))
Wmax=2*mec*np.power(beta*gamma,2)


#E1 es para PARTÍCULAS ALFA 
E1=0.15*(auz/omega)*np.power((alfaz/beta),2)*(np.log((2*mec*np.power(beta*gamma,2)*Wmax)/np.power(I,2))-2*np.power(beta,2))
Ealfa=(1./ro)*E1*10

#E2 para PROTONES

E2=0.15*(auz/omega)*np.power((proton/beta),2)*(np.log((2*mec*np.power(beta*gamma,2)*Wmax)/np.power(I,2))-2*np.power(beta,2))
Eproton=(1./ro)*E2*10

fig=plt.figure(figsize=(6.0,6.0))
plt.plot(E,Ealfa,label='Alfa')
plt.plot(E,Eproton,label='Proton')
plt.ylabel("dE/($ \\rho$dx)(MeV/($g/cm^2$))")
plt.xlabel("E(MeV)")
plt.legend(loc=0)
plt.title("Comparación de pérdida específica de la energía")
plt.yscale('log')
plt.xscale('log')
plt.show()
fig.savefig("energias")














