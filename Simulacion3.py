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
from scipy.stats import poisson

#Parámetros correspondientes al Isotopo a trabajar
#Nombre Isotopo : Carbono 14
semivida=55.68 #Unidad de tiempo :  Siglos
Gamma=np.log(2)/semivida
tau=semivida/np.log(2)

N0= 100 #Numero de nucleos iniciales
dt=1.#tau/10.
tf=3*tau
num_intervalos=round(tf/dt)

timegrid=np.arange(0,num_intervalos,dt)


#Funcion que define probabilidad de Poisson
def ProbPoisson(N0,dt):
    mu1=N0*Gamma*dt # Por definicion, ¿tendremos así mismo mu que va variando?
    mu2=N0*Gamma*30*dt #Por que 30?
    prob_ini= np.zeros(N0) 
    prob_inter=np.zeros(N0)
    for i in range(N0):
        prob_ini[i]=(np.power(mu1,i)*np.exp(-mu1))/np.math.factorial(i)#Probabilidad de que x nucleos decaiga en un tiempo inicial
        prob_inter[i]=(np.power(mu2,i)*np.exp(-mu2))/np.math.factorial(i) #Probabilidad de que decaiga en tiempo intermedio
    return prob_ini,prob_inter
muini=N0*Gamma*dt
ini,inter=ProbPoisson(N0,dt)

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


#Funcion acumulada
"""
def acumulada(N0,prob):
    F=np.zeros(N0)
    for j in range(99): 
        F[j + 1] = F[j]   +prob[j+1]     
    return F
    
#Finter=acumulada(N0,inter)
#Fini=acumulada(N0,ini)    
"""

Finter=np.cumsum(inter)
Fini=np.cumsum(ini)
plt.figure()
plt.plot(xnucleos,Finter,'r',label="Tiempo intermedio")
plt.plot(xnucleos,Fini,'b',label="Tiempo inicial")
plt.title("Probabilidad acumulada de decaimiento en 3*Tau siglos con tiempo intermedio")
plt.xlabel("Nucleos")
plt.ylabel("Probabilidad")
plt.legend(loc=0)
plt.show()

#Actividad

def actividad(N0,dt):
    ENE=[N0]
    A=[poisson.rvs(muini,size=1)]
    N0=N0-np.sum(A)
    
    while N0>0:
        ENE.append(N0)
        mu=N0*Gamma*dt
        A.append(poisson.rvs(mu,size=1))
        N0=N0-A[-1]
        #print(N0)
        
    for i in range(len(A)):
        A[i]=A[i]/dt
    return A,ENE

A,ENE=actividad(N0,dt)
plt.figure()
print(np.sum(A))
time=np.arange(0,len(A)*dt,dt)
plt.plot(time,ENE,ds="steps",label="Actividad")
plt.xlabel("Tiempo")
plt.ylabel("Nucleos")
#plt.yticks(np.arange(0,N0,10))

#Probabilidad de poisson con pmf
timegrid=np.arange(0,80,dt)
probini=poisson.pmf(timegrid,muini)
probinter=poisson.pmf(timegrid,muini*30)
plt.figure()
plt.plot(timegrid,probini)
plt.plot(timegrid,probinter)
plt.title("LA de")
plt.show()

plt.figure()
cum0=np.cumsum(probini)
cum1=np.cumsum(probinter)
plt.plot(timegrid,cum0)
plt.plot(timegrid,cum1)
plt.show()


N1=10
N2=100
N3=1000
N4=10**6


A1,ENE1=actividad(N1,dt)
A2,ENE2=actividad(N2,dt)
A3,ENE3=actividad(N3,dt)
A4,ENE4=actividad(N4,dt)
#print(np.sum(A1))
time=np.arange(0,len(A1)*dt,dt)

fig=plt.figure()

y1=(N1*np.exp(-time*Gamma))
plt.plot(time,ENE1,ds="steps",label="Actividad")
plt.plot(time,y1,label="Ley de decaimiento")
plt.title("Comparación para N=10")
plt.xlabel("Tiempo")
plt.legend(loc=0)
plt.ylabel("Nucleos")
plt.yscale('log')
plt.show()
fig=plt.figure()

print(np.sum(A1))
time=np.arange(0,len(A2)*dt,dt)

y2=(N2*np.exp(-time*Gamma))
plt.plot(time,ENE2,ds="steps",label="Actividad")
plt.plot(time,y2,label="Ley de decaimiento")
plt.title("Comparación para N=100")
plt.xlabel("Tiempo")
plt.legend(loc=0)
plt.ylabel("Nucleos")
plt.yscale('log')
plt.show()
fig=plt.figure()

time=np.arange(0,len(A3)*dt,dt)

y2=(N3*np.exp(-time*Gamma))
plt.plot(time,ENE3,ds="steps",label="Actividad")
plt.plot(time,y2,label="Ley de decaimiento")
plt.title("Comparación para N=1000")
plt.xlabel("Tiempo")
plt.legend(loc=0)
plt.ylabel("Nucleos")
plt.yscale('log')
plt.show()

fig=plt.figure()

time=np.arange(0,len(A4)*dt,dt)

y2=(N4*np.exp(-time*Gamma))
plt.plot(time,ENE4,ds="steps",label="Actividad")
plt.plot(time,y2,label="Ley de decaimiento")
plt.title("Comparación para N=10^6")
plt.xlabel("Tiempo")
plt.legend(loc=0)
plt.ylabel("Nucleos")
plt.yscale('log')
plt.show()


f=lambda x : np.exp(-x*Gamma)*Gamma
valor=integrate.quad(f,0,N0*tau)
x=np.arange(0,len(A2)*dt,dt)
y=np.zeros(len(x))
def pdf(x,y):
    for i in range(len(x)):
        #y[i]=np.exp(-(x[i]+deltat)/2*gamma)*gamma
        y[i]=(integrate.quad(lambda x: np.exp(-x*Gamma)*Gamma,x[i],x[i]+dt))[0]
    #y=  np.exp(-x*gamma)*gamma
    return y


def simulacion(N,F):#,histograma):
    histograma=np.zeros(len(F))
    quedan=np.ones(len(F))*N
    rn.seed(1)
    for i in range(N):
        r = rn.random()
        for j in range(len(histograma)):
            if r >= F[j-1] and r < F[j]:
                bingo = j
                break
        histograma[bingo]+=1
    quedan=quedan-np.cumsum(hid)
    return histograma

#Prueba exponencial
y=pdf(x,y)
F=np.cumsum(y)
expo=simulacion(N0,F)
y2=(N2*np.exp(-x*Gamma))*(np.max(A2)/N0)
print(len(F),np.sum(expo))
plt.plot(x,expo,ds='steps',label="Exponencial")
plt.plot(x,A2,ds='steps',label="Poisson")
plt.plot(x,y2,label="Ley de decaimiento")
plt.legend(loc=0)
plt.yscale('log')