# -*- coding: utf-8 -*-
"""
Created on Wed Sep 29 16:24:13 2021

@author: Andresda
"""

import numpy as np
import matplotlib.pyplot as plt

#Para cambiar en cada archivo, importa los datos
nombre1="./Na22"
nombre2="./Cs137"
nombre3="./Co60"
def general(nombre):
    
#Aqui añadimos la extension mca para importar y de una vez le doy el formato
# para al final dejarlo con el jpg
    archivo=nombre+".mca" 
    nameimagen=nombre+".jpg"
#Traer los datos al programa y hace arreglo f
    f= np.genfromtxt(archivo,dtype=int)
    y=f # Tomamos y=f por convención
    x=np.arange(1,len(y)) #Corresponde al numero de canales que hay en la medicion

    #y3=np.array(y)-np.array(y2)
    y3=y
    mu= np.mean(y)
    sigma= np.std(x)
    M= 1/(np.sqrt(2*np.pi)*sigma)

    gauss= M* np.exp(-np.power((mu-x)/sigma,2)-2)

plt.plot(x,y3)
plt.grid(True)
plt.title("Gráfica Ajustada NaI-3x3 -PX5 - CO57  ; 60s a 1000V ; Setup 2",  loc='center', pad=None)
plt.xlabel("Canal")
plt.vline(mu)
plt.ylabel("Cuentas")


nameimagen=nombre[3:]+"-Ajuste"+".jpg"

plt.savefig(nameimagen)
