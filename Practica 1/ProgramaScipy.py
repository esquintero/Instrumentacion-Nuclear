# -*- coding: utf-8 -*-
"""
Created on Wed Sep 29 16:24:13 2021

@author: Andresda
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import (AutoMinorLocator, MultipleLocator,AutoLocator)
from scipy.optimize import curve_fit
from pandas import DataFrame as df
from scipy.integrate import quad

#Para cambiar en cada archivo, importa los datos
nombre="./Na22"
#Aqui añadimos la extension para importar y de una vez le doy el formato
# para al final dejarlo con el jpg
archivo=nombre+".dat" 
nameimagen=nombre+".jpg"
#Traer los datos al programa y hace arreglo f
y= np.genfromtxt(archivo,dtype=int,usecols=(1))
 # Tomamos y=f por convención
x=np.arange(len(y)) #Corresponde al numero de canales que hay en la medicion

#y3=np.array(y)-np.array(y2)
a0=100.
a1=1.

print(y)
############################################################
############################################################
################## G R A F I C A    I N I C I A L ##########
############################################################
############################################################

#Configuración de las grillas para graficar
fig=plt.figure(figsize=(10,10))

#ax = plt.subplots()
# Grillas mayores intervalos 
#ax.xaxis.set_major_locator(AutoLocator())
#ax.yaxis.set_major_locator(AutoLocator())

# Grillas menores para determinar intervalos
#ax.xaxis.set_minor_locator(AutoMinorLocator())
#ax.yaxis.set_minor_locator(AutoMinorLocator())
#ax.grid(which='minor', color='#CCCCCC', linestyle=':')



plt.plot(x,y)

#plt.grid(True)
plt.title("Gráfica espectro Na22",  loc='center', pad=None)
plt.xlabel("Canal")
plt.ylabel("Cuentas")
#plt.figure(0)

#plt.xticks(np.arange(0,4000,100))
nameimagen=nombre[0:]+".png"

fig.savefig(nameimagen)
plt.show()


#########################################################################
######################## P I C O S  I N D I V I D U A L E S #############
#########################################################################

#Pico 1 Na22

#Se define la funcion para el ajuste gaussiano
def func(x,M,mu,sigma,a0,a1):
    return (a1*(x-mu)+a0)+M*np.exp((-1/2)*((x-mu)/sigma)**2)

#Intervalo de FORMA MANUAL para cada PICO
i1=y[1130:1430]
xi1=x[1130:1430]

#Ver ese pico
fig=plt.figure()

#fig, ax = plt.subplots()
plt.plot(xi1,i1,label="Datos")

plt.xlabel("Canal")
plt.ylabel("Cuentas")




M1=np.max(i1)
mu1=np.mean(xi1)
sigma1=np.sqrt(mu1)
centroide=xi1[0]+np.where(i1==M1)[0]
#inten=quad(func,xi1[0],xi1[-1])
inten=np.sum(i1)
na221=["Pico 1 Na22",mu1,sigma1,2.35*sigma1,inten]

##################################################################
################## N O R M A L I Z A C I Ó N #####################
##################################################################



popt, pcov = curve_fit(func, xi1, i1, p0=[M1,mu1,sigma1,a0,a1])

print(popt)


plt.plot(xi1, func(xi1, *popt),label="Ajuste")
plt.title("Gráfica Ajuste Gaussiano pico #1 Na22",  loc='center', pad=None)
plt.xlabel("Canal")
plt.ylabel("Cuentas")

plt.vlines(centroide, 0, M1,color="g",label="Centroide")
plt.legend(loc=0)

nameimagen=nombre[0:]+"-Ajuste Gaussiano"+".png"

fig.savefig(nameimagen)


#Definición de los nuevos parametros para las nuevas graficas

#########################################################################
######################## P I C O S  I N D I V I D U A L E S #############
#########################################################################

#Pico 2 Sodio 22

#Se define la funcion para el ajuste gaussiano
def func(x,M,mu,sigma,a0,a1):
    return (a1*(x-mu)+a0)+M*np.exp((-1/2)*((x-mu)/sigma)**2)

#Intervalo de FORMA MANUAL para cada PICO
i1=y[2850:3400]
xi1=x[2850:3400]

#Ver ese pico
fig=plt.figure()

#fig, ax = plt.subplots()
plt.plot(xi1,i1,label="Datos")

plt.xlabel("Canal")
plt.ylabel("Cuentas")




M1=np.max(i1)
mu1=np.mean(xi1)

sigma1=np.sqrt(mu1)
centroide=xi1[0]+np.where(i1==M1)[0]
#inten2=quad(func,xi1[0],xi1[-1])
inten=np.sum(i1)
na222=["Pico 2 Na22",mu1,sigma1,2.35*sigma1,inten]
##################################################################
################## N O R M A L I Z A C I Ó N #####################
##################################################################


popt, pcov = curve_fit(func, xi1, i1, p0=[M1,mu1,sigma1,a0,a1])

#print(popt)


plt.plot(xi1, func(xi1, *popt),label="Ajuste")
plt.title("Gráfica Ajuste Gaussiano pico #2 Na22",  loc='center', pad=None)
plt.xlabel("Canal")
plt.ylabel("Cuentas")

plt.vlines(centroide, 0, M1,color="g",label="Centroide")
plt.legend(loc=0)

nameimagen=nombre[0:]+"-Ajuste Gaussiano2"+".png"

fig.savefig(nameimagen)



#######################################################################
#######################################################################
#######################################################################



#Para cambiar en cada archivo, importa los datos
nombre="./Cs137"
#Aqui añadimos la extension para importar y de una vez le doy el formato
# para al final dejarlo con el jpg
archivo=nombre+".dat" 
nameimagen=nombre+".jpg"
#Traer los datos al programa y hace arreglo f
y= np.genfromtxt(archivo,dtype=int,usecols=(1))
 # Tomamos y=f por convención
x=np.arange(len(y)) #Corresponde al numero de canales que hay en la medicion

#y3=np.array(y)-np.array(y2)
a0=100.
a1=1.

print(y)
############################################################
############################################################
################## G R A F I C A    I N I C I A L ##########
############################################################
############################################################



fig=plt.figure(figsize=(10,10))



plt.plot(x,y)

#plt.grid(True)
plt.title("Gráfica espectro Cs137",  loc='center', pad=None)
plt.xlabel("Canal")
plt.ylabel("Cuentas")
#plt.figure(0)

#plt.xticks(np.arange(0,4000,100))
nameimagen=nombre[0:]+".png"

fig.savefig(nameimagen)
plt.show()


#########################################################################
######################## P I C O S  I N D I V I D U A L E S #############
#########################################################################

#Pico 1 Na22

#Se define la funcion para el ajuste gaussiano
def func(x,M,mu,sigma,a0,a1):
    return (a1*(x-mu)+a0)+M*np.exp((-1/2)*((x-mu)/sigma)**2)

#Intervalo de FORMA MANUAL para cada PICO
i1=y[60:125]
xi1=x[60:125]

#Ver ese pico
fig=plt.figure()

#fig, ax = plt.subplots()
plt.plot(xi1,i1,label="Datos")

plt.xlabel("Canal")
plt.ylabel("Cuentas")




M1=np.max(i1)
mu1=np.mean(xi1)
sigma1=np.sqrt(mu1)
centroide=xi1[0]+np.where(i1==M1)[0]
#inten=quad(func,xi1[0],xi1[-1])
inten=np.sum(i1)
cs1371=["Pico 1 Cs137",mu1,sigma1,2.35*sigma1,inten]

##################################################################
################## N O R M A L I Z A C I Ó N #####################
##################################################################



popt, pcov = curve_fit(func, xi1, i1, p0=[M1,mu1,sigma1,a0,a1])

#print(popt)


plt.plot(xi1, func(xi1, *popt),label="Ajuste")
plt.title("Gráfica Ajuste Gaussiano pico #1 Cs137",  loc='center', pad=None)
plt.xlabel("Canal")
plt.ylabel("Cuentas")

plt.vlines(centroide, 0, M1,color="g",label="Centroide")
plt.legend(loc=0)

nameimagen=nombre[0:]+"-Ajuste Gaussiano"+".png"

fig.savefig(nameimagen)


#Definición de los nuevos parametros para las nuevas graficas

#########################################################################
######################## P I C O S  I N D I V I D U A L E S #############
#########################################################################

#Pico 2 Cesio 137

#Se define la funcion para el ajuste gaussiano
def func(x,M,mu,sigma,a0,a1):
    return (a1*(x-mu)+a0)+M*np.exp((-1/2)*((x-mu)/sigma)**2)

#Intervalo de FORMA MANUAL para cada PICO
i1=y[1400:1900]
xi1=x[1400:1900]

#Ver ese pico
fig=plt.figure()

#fig, ax = plt.subplots()
plt.plot(xi1,i1,label="Datos")

plt.xlabel("Canal")
plt.ylabel("Cuentas")




M1=np.max(i1)
mu1=np.mean(xi1)

sigma1=np.sqrt(mu1)
centroide=xi1[0]+np.where(i1==M1)[0]
#inten2=quad(func,xi1[0],xi1[-1])
inten=np.sum(i1)
cs1372=["Pico 2 Cs137",mu1,sigma1,2.35*sigma1,inten]
##################################################################
################## N O R M A L I Z A C I Ó N #####################
##################################################################


popt, pcov = curve_fit(func, xi1, i1, p0=[M1,mu1,sigma1,a0,a1])

#print(popt)


plt.plot(xi1, func(xi1, *popt),label="Ajuste")
plt.title("Gráfica Ajuste Gaussiano pico #2 Cs137",  loc='center', pad=None)
plt.xlabel("Canal")
plt.ylabel("Cuentas")

plt.vlines(centroide, 0, M1,color="g",label="Centroide")
plt.legend(loc=0)

nameimagen=nombre[0:]+"-Ajuste Gaussiano2"+".png"

fig.savefig(nameimagen)



datos=[na221,na222,cs1371,cs1372]
#print(df(datos,columns=["Dato","\mu","\sigma","FWHM","Intensidad"]))
#Definición de los nuevos parametros para las nuevas graficas



#######################################################################
#######################################################################
#######################################################################



#Para cambiar en cada archivo, importa los datos
nombre="./Co60"
#Aqui añadimos la extension para importar y de una vez le doy el formato
# para al final dejarlo con el jpg
archivo=nombre+".dat" 
nameimagen=nombre+".jpg"
#Traer los datos al programa y hace arreglo f
y= np.genfromtxt(archivo,dtype=int,usecols=(1))
 # Tomamos y=f por convención
x=np.arange(len(y)) #Corresponde al numero de canales que hay en la medicion

#y3=np.array(y)-np.array(y2)
a0=100.
a1=1.

print(y)
############################################################
############################################################
################## G R A F I C A    I N I C I A L ##########
############################################################
############################################################



fig=plt.figure(figsize=(10,10))



plt.plot(x,y)

#plt.grid(True)
plt.title("Gráfica espectro Co60",  loc='center', pad=None)
plt.xlabel("Canal")
plt.ylabel("Cuentas")
#plt.figure(0)

#plt.xticks(np.arange(0,4000,100))
nameimagen=nombre[0:]+".png"

fig.savefig(nameimagen)
plt.show()


#########################################################################
######################## P I C O S  I N D I V I D U A L E S #############
#########################################################################

#Pico 1 Co60

#Se define la funcion para el ajuste gaussiano
def func(x,M,mu,sigma,a0,a1):
    return (a1*(x-mu)+a0)+M*np.exp((-1/2)*((x-mu)/sigma)**2)

def func1(x,M1,mu1,M2,mu2,sigma1,sigma2):
    return M1*np.exp((-1/2)*((x-mu1)/sigma1)**2)+M2*np.exp((-1/2)*((x-mu2)/sigma2)**2)

#Intervalo de FORMA MANUAL para cada PICO
i1=y[2500:3080]
xi1=x[2500:3080]

#Ver ese pico







M1=np.max(i1)
mu1=np.mean(xi1)
sigma1=np.sqrt(mu1)
centroide1=xi1[0]+np.where(i1==M1)[0]
#inten=quad(func,xi1[0],xi1[-1])
inten=np.sum(i1)
co601=["Pico 1 Co60",mu1,sigma1,2.35*sigma1,inten]

i1=y[3080:3600]
xi1=x[3080:3600]


#Ver ese pico
fig=plt.figure()

#fig, ax = plt.subplots()



M2=np.max(i1)
mu2=np.mean(xi1)

sigma2=np.sqrt(mu2)
centroide2=xi1[0]+np.where(i1==M2)[0]
#inten2=quad(func,xi1[0],xi1[-1])
inten=np.sum(i1)
co602=["Pico 2 Co60",mu2,sigma2,2.35*sigma2,inten]

plt.plot(x[2500:3600],y[2500:3600],label="Datos")

plt.xlabel("Canal")
plt.ylabel("Cuentas")


##################################################################
################## N O R M A L I Z A C I Ó N #####################
##################################################################



popt, pcov = curve_fit(func1,x[2500:3600] , y[2500:3600], p0=[M1,mu1,M2,mu2,sigma1,sigma2])

#print(popt)


plt.plot(x[2500:3600], func1(x[2500:3600],*popt),label="Ajuste")
plt.title("Gráfica Ajuste Gaussiano Co60",  loc='center', pad=None)
plt.xlabel("Canal")
plt.ylabel("Cuentas")
plt.vlines([centroide1,centroide2],[[ 0,0]], [M1,M2],color="g",label="Centroide")
plt.legend(loc=0)

nameimagen=nombre[0:]+"-Ajuste Gaussiano"+".png"

fig.savefig(nameimagen)

datos=[na221,na222,cs1371,cs1372,co601,co602]
print(df(datos,columns=["Dato","$\mu$","$\sigma$","FWHM","Intensidad"]))

