# -*- coding: utf-8 -*-
"""
Created on Mon Feb  7 13:41:37 2022

@author: Andresda
"""


import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import random as rn
import pandas as pd
from scipy.optimize import curve_fit
from scipy.integrate import quad

import matplotlib as mpl
mpl.rcParams['legend.fontsize'] = 20
mpl.rcParams['axes.labelsize'] = 20
mpl.rcParams['xtick.labelsize'] = 20
mpl.rcParams['ytick.labelsize'] = 20
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['mathtext.fontset'] = 'dejavusans'
mpl.rcParams.update({'font.size': 10})



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
 

#Parámetros a usar en la simulación

muc=0.2504			    #Compton
muf=0.0396				#Fotoeléctrico
musuma= muc+muf         #Mu suma
coc= 1/musuma           #Cociente de sumas de mu
t=7.62                  #longitud del detector en centimetros  


#Vamos a realizar la funcion si es o no es detectado
F1=[np.exp(-1*muc*t),1-np.exp(-1*musuma*t)]
F1=np.cumsum(F1)
hist=simulacion(10,F1)[0] #Contiene en la posicion 0 cuantos son detectados
N=hist[0]
N=int(N)
plt.figure()
plt.plot(['Decae','No decae'],hist,ds='steps-mid')

#Ahora con los datos detectados veremos si decae en F o C

F2=[muf*coc,muc*coc]
F2=np.cumsum(F2)
detec=simulacion(N,F2)[0]
print(detec)

plt.figure()
plt.plot(['Fotoeléctrico','Compton'],detec,ds='steps-mid')

    
  