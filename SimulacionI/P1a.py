# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 16:19:40 2021

@author: Esteban
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import random as rn

fmax = 1/6
grid = np.zeros(12)
m1 = fmax/8
m2 = -fmax/4

pdf = np.zeros(12)
F = np.zeros(12)

#  Funcion densidad de probabilidad

for i in range(12):
    grid[i] = 0.5 + i
    if i <= 7:
        pdf[i] = m1*grid[i]
      
    else:
        pdf[i] = m2*grid[i] - m2*12

        
# Acumulativa

for j in range(11): 
    F[j + 1] = F[j] + pdf[j + 1]        
       
    

#simulación
# Número de historias = número de días
N = 30
#histograma=np.arange(6,18,1)

#rn.seed(12345)
def simulacion(N):#,histograma):
    histograma=np.arange(6,18,1)
    for i in range(N):
        r = rn.random()
        for j in range(len(histograma)):
            if r >= F[j-1] and r < F[j]:
                bingo = j
                break
        histograma[bingo]+=1
    return histograma
    
    

#print(histograma)

plt.figure(figsize=(6.0,6.0))
ax=plt.gca()
plt.plot(grid, pdf, 'bo')
plt.title("Probabilidad de que llueva en Bogotá")
plt.xlabel("Tiempo t(hora)")
plt.ylabel("Probabilidad de lluvia P(t)")
plt.grid(True)

#Grafica de Funcion acumulada
plt.figure(figsize=(5.0,4.8))
plt.plot(grid, F, ds='steps-mid',c='r')
plt.title("Funcion acumulativa ")
plt.xlabel("Tiempo t (hora)")
plt.ylabel("F(t)")
ax=plt.gca()

ax.set_xticks(range(0,12))
plt.grid(True)

#Grafica de Probabilidad = 30
plt.figure(figsize=(5.0,4.8))
plt.plot(grid, simulacion(30), ds='steps-mid',c='r')
plt.title("Frecuencia de lluvia en 30 días")
plt.xlabel("Tiempo t (hora)")
plt.ylabel("Frecuencia")
ax=plt.gca()

ax.set_xticks(range(0,12))
plt.grid(True)


#Grafica de Probabilidad = 365
plt.figure(figsize=(5.0,4.8))
plt.plot(grid, simulacion(365), ds='steps-mid',c='r')
plt.title("Frecuencia de lluvia en 365 días")
plt.xlabel("Tiempo t (hora)")
plt.ylabel("Frecuencia")
ax=plt.gca()

ax.set_xticks(range(0,12))
plt.grid(True)


#Grafica de Probabilidad = 1000
plt.figure(figsize=(5.0,4.8))
plt.plot(grid, simulacion(1000), ds='steps-mid',c='r')
plt.title("Frecuencia de lluvia en 1000 días")
plt.xlabel("Tiempo t (hora)")
plt.ylabel("Frecuencia")
ax=plt.gca()

ax.set_xticks(range(0,12))
plt.grid(True)


#Grafica de Probabilidad = 10^6
plt.figure(figsize=(5.0,4.8))
plt.plot(grid, simulacion(np.power(10,6)), ds='steps-mid',c='r')
plt.title("Frecuencia de lluvia en 10^6 días")
plt.xlabel("Tiempo t (hora)")
plt.ylabel("Frecuencia")
ax=plt.gca()

ax.set_xticks(range(0,12))
plt.grid(True)
plt.show()

