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

#Ejecutamos las funciones y asignar valor mu y sigma
mu=valmedio(x)
sigma=desv(x)
print(mu,sigma)
"""
def contadorPrueba(x,delta):
    
    rango=np.arange(np.min(x),np.max(x),delta)
    histograma=np.zeros(len(rango))
    x=np.sort(x)
    histograma,rango=plt.hist(x,rango)
    return histograma, rango
"""           
#Función para agrupar datos en rangos para hacer el histograma         
def contador(x,delta):
    
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

x1,y1=contador(x,2.5)
x2,y2=contador(x,10)
"""
plt.figure()
plt.plot(x1,y1,ds='steps')
plt.title("Frecuencia cada 2.5 mV")
plt.ylabel("Frecuencia")
plt.xlabel("Voltaje (mV)")


plt.figure()
plt.plot(x2,y2,ds='steps')
plt.title("Frecuencia cada 10 mV")
plt.ylabel("Frecuencia")
plt.xlabel("Voltaje (mV)")
"""

# Definicoin de funcion para valor medio de los histogramas
def valmedio2(x):
    suma=0
    for i in range(len(x)):
        suma+=((x[i-1]+x[i])/2)*x[i]
        
    mu=suma/np.sum(x)
    return mu
mu2=valmedio2(x1)
mu3=valmedio2(x2)

#Función para hacer la varianza y así determinar la desviación de los histogramas
def desv2(x,mu):
    suma=0
    for i in range(len(x)):
        suma+=np.power((((x[i-1]+x[i])/2)-mu),2)*x[i]
        
    sigmaq=suma/np.sum(x)
    sigmaq=np.sqrt(sigmaq)
    return sigmaq
sigma2=desv2(x1,mu2)
sigma3=desv2(x2,mu3)

  


"""
ax,fig=plt.subplots(2,1)

fig[0].plot(x1,y1,ds='steps')
fig[1].plot(x2,y2,ds='steps')
ax[1].yticks(range(0,50))
fig[1].title("Datos cada 10 mV")
"""



