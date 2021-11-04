# -*- coding: utf-8 -*-
"""
Created on Thu Nov  4 14:01:20 2021

@author: Andresda
"""
import numpy as np
import matplotlib.pyplot as plt

#Importar el archivo .dat
archivo="amplificador.dat"
x= np.genfromtxt(archivo,dtype=int, skip_header=1)

#Funcion para el valor medio
def valmedio(x):
    mu=np.sum(x)/200
    return mu

#Valores obtenidos con numpy para media y desviacion para corroborar
mu1=np.mean(x)
sigma1=np.std(x)



#Determinar la desviacion estándar
def desv(x):
    suma=0
    for i in range(len(x)):
        suma+=np.power(mu-x[i],2)
    sigma=np.sqrt(suma/200.)
    return sigma    

mu=valmedio(x)
sigma=desv(x)
print(mu,sigma)
"""
def contador(x,delta):
    
    rango=np.arange(np.min(x),np.max(x),delta)
    histograma=np.zeros(len(rango))
    x=np.sort(x)
    histograma,rango=plt.hist(x,rango)
    return histograma, rango
"""           
         
def contador2(x,delta):
    
    rango=np.arange(np.min(x),np.max(x),delta)
    histograma=np.zeros(len(rango))
    x=np.sort(x)
    
    for j in range (len(histograma)):
        sumar=0
        for i in range(len(x)):
            
            if x[i]>=rango[j-1] and x[i]<rango[j]:
                sumar+=1
        histograma[j]=sumar
    return rango, histograma

x1,y1=contador2(x,2.5)
#print()
plt.plot(x1,y1,ds='steps-mid')
