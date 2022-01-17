# -*- coding: utf-8 -*-
"""
Created on Wed Sep 29 16:24:13 2021

@author: Andresda
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from matplotlib.ticker import (AutoMinorLocator, MultipleLocator,AutoLocator)
from pandas import DataFrame as df
from scipy.integrate import quad

import matplotlib as mpl
mpl.rcParams['legend.fontsize'] = 20
mpl.rcParams['axes.labelsize'] = 20
mpl.rcParams['xtick.labelsize'] = 20
mpl.rcParams['ytick.labelsize'] = 20
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['mathtext.fontset'] = 'dejavusans'
mpl.rcParams.update({'font.size': 10})


from matplotlib.ticker import AutoMinorLocator
from matplotlib.ticker import NullFormatter


def cm2inch(cm):
    return cm/2.54

def gaussBckgrnd(x,m,s,M,b0,b1):
    z = (m-x)/s
    gauss = M*np.exp(-0.5*z**2)
    fondo = b0 + b1 * (x - m)
    return gauss + fondo

def tandem(x,m1,s1,M1,m2,s2,M2,b0,b1):
    z1 = (m1-x)/s1
    gauss1 = M1*np.exp(-0.5*z1**2)
    z2 = (m2-x)/s2
    gauss2 = M2*np.exp(-0.5*z2**2)
    mm = (m1 + m2)/2
    fondo = b0 + b1 * (x - mm)
    return gauss1 + gauss2 + fondo

def recta(x,b0,b1,x0):
    return b0 + b1 * (x-x0)

def gauss(x,m,s,M):
    z = (m-x)/s
    return M*np.exp(-0.5*z**2)

# toma las salidas de curve_fit: parámetros= pot, matriz de covariaza = pcov
# La diagonal de la matriz de covarianza es la covarianza de cada parámetro. Su raíz cuadrada es la desviación estándar.
def statsGauss_tabla(popt,pcov):
    mu = popt[0] 
    stdv_mu = np.sqrt(pcov[0,0])
    
    s = popt[1]
    s = np.abs(s) # curve_fit puede producir s < 0, sin efecto adverso. 
    stdv_s = np.sqrt(pcov[1,1])
    
    M = popt[2]
    stdv_M = np.sqrt(pcov[2,2])
    
    b0 = popt[3]
    stdv_b0 = np.sqrt(pcov[3,3])
    
    b1 = popt[4]
    stdv_b1 = np.sqrt(pcov[4,4])
    
    I = np.sqrt(2*np.pi)*s*M
    stdv_I = I * np.sqrt((stdv_s/s)**2 + (stdv_M/M)**2) 

    print('   M(c/cnl)      mu(cnl)     sigma(cnl)   I(c) ')
    print('{0:4.2f}({1:2.2f}) &{2:4.1f}({3:2.1f}) &{4:3.1f}({5:2.1f}) &{6:5.0f}({7:3.0f})  \\\\'.format(M,stdv_M,mu,stdv_mu,s,stdv_s,I,stdv_I))



######################################################################
######################################################################
######################################################################


y0= np.genfromtxt('./datos/NaI_Fondo_600s.dat',dtype=int,usecols=(1),skip_header=1)
y1= np.genfromtxt('./datos/NaI_22Na_600s.dat',dtype=int,usecols=(1),skip_header=1)
y2= np.genfromtxt('./datos/NaI_60Co_600s.dat',dtype=int,usecols=(1),skip_header=1)
y3= np.genfromtxt('./datos/NaI_137Cs_600s.dat',dtype=int,usecols=(1),skip_header=1)
y4= np.genfromtxt('./datos/NaI_Fondo_300s.dat',dtype=int,usecols=(1),skip_header=1)   
y5= np.genfromtxt('./datos/NaI_57Co_300s.dat',dtype=int,usecols=(1),skip_header=1)
x0 = np.arange(0,len(y0))

#Limitar el tamaño de los datos de la gráfica


fig1=plt.figure(figsize=(12,6))
plt.plot(x0,y0,drawstyle='steps-mid',linewidth=2,color='r',label=r'Fondo 600s')
plt.plot(x0,y1,drawstyle='steps-mid',linewidth=2,label=r'$^{22}$Na')
plt.plot(x0,y2,drawstyle='steps-mid',linewidth=2,label=r'$^{60}$Co')
plt.plot(x0,y3,drawstyle='steps-mid',linewidth=2,label=r'$^{137}$Cs')
plt.plot(x0,y4,drawstyle='steps-mid',linewidth=2,label=r'Fondo 300s')
plt.plot(x0,y5,drawstyle='steps-mid',linewidth=2,label=r'$^{57}$Co')
plt.xlabel("Canal")
plt.ylabel("Cuentas/Canal")
plt.legend()
plt.title("Espectros con detector NaI 3x3")
plt.show()
#fig1.savefig("espectrosjuntos.png")

y1=np.array(y1)-np.array(y0)
y2=np.array(y2)-np.array(y0)
y3=np.array(y3)-np.array(y0)
y5=np.array(y5)-np.array(y4)

fig2=plt.figure(figsize=(12,6))
plt.plot(x0,y1,drawstyle='steps-mid',linewidth=2,label=r'$^{22}$Na')
plt.plot(x0,y2,drawstyle='steps-mid',linewidth=2,label=r'$^{60}$Co')
plt.plot(x0,y3,drawstyle='steps-mid',linewidth=2,label=r'$^{137}$Cs')
plt.plot(x0,y5,drawstyle='steps-mid',linewidth=2,label=r'$^{57}$Co')
plt.legend()
plt.xlabel("Canal")
plt.ylabel("Cuentas/Canal")
plt.title("Espectros con detector NaI 3x3 sin Fondo")
plt.show()
fig2.savefig("espectrosinfondo")


#Cobalto 60
plt.figure(figsize=(12,6))
plt.plot(x0,y2,drawstyle='steps-mid',linewidth=2,label=r'$^{60}$Co')
plt.legend()
plt.title("Espectro $^{60}$Co ")
plt.show()

"""
#Sodio
plt.figure(figsize=(12,6))
plt.plot(x0,y1,drawstyle='steps-mid',linewidth=2,label=r'$^{22}$Na')
plt.legend()
plt.title("Espectro $^{22}$Na")
plt.show()
plt.savefig("na22.png")



#Cesio 137
plt.figure(figsize=(12,6))
plt.plot(x0,y3,drawstyle='steps-mid',linewidth=2,label=r'$^{137}$Cs')
plt.legend()
plt.title("Espectro $^{137}$Cs")
plt.show()

#Cesio 137
plt.figure(figsize=(12,6))
plt.plot(x0,y5,drawstyle='steps-mid',linewidth=2,label=r'$^{57}$Co')
plt.legend()
plt.title("Espectro $^{57}$Co")
plt.show()
"""
"""
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
#inten=quad(func,1130,1430,(popt[0],popt[1],popt[2],popt[3],popt[4]))[0]

#na221=["Pico 1 Na22",popt[1],popt[2],2.35*popt[2],inten]
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
"""


"""

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
"""
"""
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


#######################################################################
########################  C A L I B R A C I O N #######################
#######################################################################



# Imprimimos la tabla con los datos a comparar

datos=[na221,na222,cs1371,cs1372,co601,co602]   
tabla=df(datos,columns=["Dato","$\mu$","$\sigma$","FWHM","Intensidad"])
print(tabla)

# Creamos el vector de valores medios sin el segundo pico de NA22
vecmu=np.zeros(6)
for i in range (6):
    vecmu[i]=datos[i][1]
#vecmu[1]=datos[5][1]

#Lo ordenamos de menor a mayor y traemos los datos suministrados por el profesor
vecmu=[1289.8,3130.4,92.3,1660.0,]
vecmu.sort() 

vecteo=[32,511,661.66,1173.23,1274.54,1332.490]
a0,a1=100,1.
#Función para el cambio a energía
def recta(x,a0,a1):
    return(a0+a1*x)

popt,pcov=curve_fit(recta,vecmu,vecteo,p0=(a0,a1))
a0,a1=popt[0],popt[1]
print(a0,a1)

plt.figure(figsize=(8.,8.))
plt.plot(vecmu,vecteo,'o',label="Valores $\mu$")
plt.plot(np.arange(0,3500), recta( np.arange(0,3500),a0,a1),label="E=-6.61+0.41*Canal")
#plt.plot(x,recta(a0,a1,x))
plt.legend(loc = 2)
plt.title("Gráfica Ajuste Energía vs Canal")
plt.xlabel("Canal")
plt.ylabel("Energía(keV)")
plt.savefig("Grafica ajuste Energía vs Canal")

print(df([[a0,a1]],columns=["a0","a1"]))


####################################################################
####################################################################
##################### T A B L A  3 #################################
####################################################################
####################################################################
vecajuste=recta(vecmu,a0,a1)
delta=np.array(vecteo)-np.array(vecajuste)
porcentaje=np.abs((delta/vecteo)*100)
#print(porcentaje)
datatable3={"E_gamma":vecteo,"Canal":vecmu,"E-Ajuste":vecajuste,"Delta":delta,"Porcentaje": porcentaje}
tabla3=df(datatable3)#columns=["$E_gamma$","Canal","E-Ajuste","$\Delta$","Porcentaje"])
print(tabla3)




"""








