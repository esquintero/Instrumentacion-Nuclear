# -*- coding: utf-8 -*-
"""
Created on Mon Nov 15 21:42:23 2021

@author: Andresda
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import random as rn
import math as math

#Parámetros correspondientes al Isotopo a trabajar
#Nombre Isotopo : Carbono 14
semivida=55.68 #Unidad de tiempo :  Siglos
Gamma=np.log(2)/semivida
tau=semivida/np.log(2)

N0= 100 #Numero de nucleos iniciales
dt=1.#tau/10.
tf=3*tau
num_intervalos=round(tf/dt)

timegrid=np.arange(1,num_intervalos,dt)


#Funcion que define probabilidad de Poisson
def ProbPoisson(Gamma,N0,dt):
    mu1=N0*Gamma*dt
    mu2=N0*Gamma*30*dt #Por que 30?
    prob_ini= np.zeros(N0) #Probabilidad de que x nucleos decaiga en un tiempo inicial
    prob_inter=np.zeros(N0) #Probabilidad de que decaiga en tiempo intermedio
    for i in range(N0):
        prob_ini[i]=(np.power(mu1,i)*np.exp(-mu1))/np.math.factorial(i)
        prob_inter[i]=(np.power(mu2,i)*np.exp(-mu2))/np.math.factorial(i)
    return prob_ini,prob_inter

ini,inter=ProbPoisson(Gamma,N0,dt)
print(ini)

xnucleos=np.arange(0,N0,1)
plt.figure()
plt.plot(xnucleos,ini,'b',label="Tiempo inicial")
plt.plot(xnucleos,inter,'r',label="Tiempo intermedio")
plt.legend(loc=0)
plt.xlabel("X Nucleos")
plt.ylabel("Probabilidad de que x nucleo decaiga")
plt.title("Probabilidad de que X nucleos decaigan")
plt.xticks(np.arange(0,100,10))
plt.show()











