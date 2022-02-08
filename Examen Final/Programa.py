# -*- coding: utf-8 -*-
"""
Created on Mon Feb  7 13:41:37 2022

@author: Andresda
"""


import numpy as np
import matplotlib.pyplot as plt
import random as rn
from pandas import DataFrame as df



#Funciones a usar en el programa

rn.seed(1234) #Semilla de números aleatorios
def simulacion(N,F):#,histograma):
    histograma=np.zeros(2)
    for i in range(N):
        r = rn.random()        
        for j in range(len(histograma)):
            if r >= F[0] and r < F[1]:
                bingo = j
                break
            else:
                bingo=j
        histograma[bingo]+=1
    return histograma,N



def simulacion2(N,F):#,histograma):
    histograma=np.zeros(477)
    for i in range(N):
        r = rn.random()
        for j in range(len(histograma)):
            if r >= F[j-1] and r < F[j]:
                bingo = j
                break
        histograma[bingo]+=1
    return histograma

def simulacion3(N,F):#,histograma):
    histograma=np.zeros(1000)
    for i in range(N):
        r = rn.random()
        for j in range(477):
            if r >= F[j-1] and r < F[j]:
                bingo = j
                bingo=np.random.default_rng().normal(bingo,float(np.abs(sigma(bingo))),1)
                bingo=int(bingo)
                break
        histograma[bingo]+=1
    return histograma


 
def sigma(x):
    return -67+2.12*np.sqrt(x)/2.35



#Parámetros a usar en la simulación

muc=0.2402382			#Compton
muf=0.03135648			#Fotoeléctrico
musuma= muc+muf         #Mu suma
coc= 1/musuma           #Cociente de sumas de mu
t=7.62                  #longitud del detector en centimetros  
r_e=2.8179*(10**-15)    #Radio clásico del electrón
numerodeentrada=10**6

#Vamos a usar la funcion si es o no es detectado
F1=[np.exp(-1*muc*t),1-np.exp(-1*musuma*t)]
F1=np.cumsum(F1)
hist=simulacion(numerodeentrada,F1)[0] #Contiene en la posicion 0 cuantos son detectados
N=hist[0]
N=int(N)
plt.figure()
plt.plot(['Decae','No decae'],hist,ds='steps-mid')

#Ahora con los datos detectados veremos si interactua en F o C

F2=[muf*coc,muc*coc]
F2=np.cumsum(F2)
detec=simulacion(N,F2)[0]
print(detec)

plt.figure()
plt.plot(['Fotoeléctrico','Compton'],detec,ds='steps-mid')

    
#Región compton
detecComp=int(detec[1])
epsilon_gamma = 662/511
bordeCompton = 662*2*epsilon_gamma/(1+2*epsilon_gamma)
Ee = np.arange(1,478)
fondoCompton = np.zeros(len(Ee))
A = np.pi*(r_e**2)/(511*epsilon_gamma**2)
for i in range(len(Ee)):
  epsilon_e = Ee[i]/662
  B = (epsilon_e**2)/((epsilon_gamma**2)*(1-epsilon_e)**2)
  C = epsilon_e/(1-epsilon_e)
  D = epsilon_e-(2/epsilon_gamma)
  fondoCompton[i] = A*(2+B+C*D)


F3=np.cumsum(fondoCompton)
F3=np.array(F3)/F3[-1]
print(F3[-1])

distcompton=simulacion2(detecComp,F3)

#Construcción del espectro uniendo Compton y Fotopico

x=np.arange(0,700)
y=np.zeros(len(x))
for i in range(len(distcompton)):
    y[i]=distcompton[i]
y[662]=detec[0]    


#Graficas para evidenciar funcionamiento
plt.figure(figsize=(8,6))
plt.plot(x,y)
plt.yscale('log')
print('comp',detec[1],'foto',detec[0])

#exportamos los datos de la simulación de 10**6 para graficar en origin
d={'Canal':x, 'Cuentas':y}
d=df(data=d)
d.to_csv('millon.txt', sep=',',index=None)


#Distribución Gaussiana de Fotoelectrico con números aleatorios de distribución normal
histograma=np.zeros(1000)

sigma1=np.abs(sigma(662))
num=detec[0]
print(num)
sf=np.random.default_rng().normal(662.,float(sigma1),int(num)) #Genera un total de numeros aleatorios basado en los que fueron detectados como fotoelectrico
for i in range(len(sf)):
    sf[i]=round(sf[i],0)
    bingo=sf[i]
    bingo=int(bingo)
    histograma[bingo]+=1
    
#Distribución ahora agregando la incertidumbre normal a la región compton
simcog=simulacion3(int(detec[1]), F3) 

#Se construye el histograma con estos datos obtenidos    
histograma=np.array(histograma)+np.array(simcog)
x=np.arange(1000)

#Exportamos los datos la simulación con incertidumbre para graficar en origin 
d={'Energía':x, 'Cuentas/canal':histograma}
d=df(data=d)
d.to_csv('gauss.txt', sep=',',index=None)

plt.figure()
plt.plot(x,histograma)
plt.yscale('log')











