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
    c=[round(mu,2),round(stdv_mu,2),round(s,2),round(stdv_s,2),round(I,2),round(stdv_I,2)]
    return c 


######################################################################
######################################################################
######################################################################

b0,b1=1.,10.

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
#fig2.savefig("espectrosinfondo")
"""


co57: 0-40 pic2 40-70
"""
"""
#Cobalto 60
plt.figure(figsize=(12,6))
plt.plot(x0,y2,drawstyle='steps-mid',linewidth=2,label=r'$^{60}$Co')
plt.legend()
plt.title("Espectro $^{60}$Co ")
plt.show()


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

#########################################################################
######################## P I C O S  I N D I V I D U A L E S #############
#########################################################################

#Pico 1 Na22

#sodio22: pico 1 150-200, pico 2 380-430
#Intervalo de FORMA MANUAL para cada PICO
i1=y1[160:200]
xi1=x0[160:200]

M1=np.max(i1)

mu=xi1[0]+np.where(i1==M1)[0]
mu=float(mu)
s1=np.sqrt(mu)

popt, pcov = curve_fit(gaussBckgrnd,xi1,i1,p0=[mu,s1,M1,b0,b1])#sigma=np.sqrt(i1))

##################################################################
################## N O R M A L I Z A C I Ó N #####################
##################################################################

mu,sigma1,M1,b0,b1 = popt
y_fit = gaussBckgrnd(xi1,mu,sigma1,M1,b0,b1)
r_fit = recta(xi1,b0,b1,mu)
plt.figure(figsize=(cm2inch(45.0),cm2inch(25.0)))
plt.plot(xi1,i1,drawstyle='steps-mid',linewidth=3,color=(0.7,0,0,0.9))
plt.plot(xi1,y_fit,lw=3)
plt.plot(xi1,r_fit)
plt.xlabel("Canal")
plt.ylabel("Cuentas/Canal")

plt.fill_between(xi1,r_fit,y_fit,color=(0,0,0.7,0.2));
na221=statsGauss_tabla(popt,pcov) # La salida produce resultados listos para incluir y editar en latex. Pero falta depurar la notación de las incertidumbres...
na221.append(round(na221[4]*2.35,2))
na221.insert(0,"Na22 Pico 1")
#########################################################################
######################## P I C O S  I N D I V I D U A L E S #############
#########################################################################

#Pico 2 Na22

#sodio22: pico 1 150-200, pico 2 380-430
#Intervalo de FORMA MANUAL para cada PICO
i1=y1[380:455]
xi1=x0[380:455]

M1=np.max(i1)

mu=xi1[0]+np.where(i1==M1)[0]
mu=float(mu)
s1=np.sqrt(mu)

popt, pcov = curve_fit(gaussBckgrnd,xi1,i1,p0=[mu,s1,M1,b0,b1])#sigma=np.sqrt(i1))

##################################################################
################## N O R M A L I Z A C I Ó N #####################
##################################################################

mu,sigma1,M1,b0,b1 = popt
y_fit = gaussBckgrnd(xi1,mu,sigma1,M1,b0,b1)
r_fit = recta(xi1,b0,b1,mu)
plt.figure(figsize=(cm2inch(45.0),cm2inch(25.0)))
plt.plot(xi1,i1,drawstyle='steps-mid',linewidth=3,color=(0.7,0,0,0.9))
plt.plot(xi1,y_fit,lw=3)
plt.plot(xi1,r_fit)
plt.xlabel("Canal")
plt.ylabel("Cuentas/Canal")

plt.fill_between(xi1,r_fit,y_fit,color=(0,0,0.7,0.2));
na222=statsGauss_tabla(popt,pcov) # La salida produce resultados listos para incluir y editar en latex. Pero falta depurar la notación de las incertidumbres...
na222.append(round(na222[4]*2.35,2))
na222.insert(0,"Na22 Pico 2")

#cobalto 60 :pico 1 350-400 , pico 2 400-460
##################################################################
################## C O 6 0                  ######################
##################################################################


#########################################################################
######################## P I C O S  I N D I V I D U A L E S #############
#########################################################################

#Pico 1 Co60

#Intervalo de FORMA MANUAL para cada PICO
i1=y2[350:410]
xi1=x0[350:410]

M1=np.max(i1)

mu=xi1[0]+np.where(i1==M1)[0]
mu=float(mu)
s1=np.sqrt(mu)

popt, pcov = curve_fit(gaussBckgrnd,xi1,i1,p0=[mu,s1,M1,b0,b1])#sigma=np.sqrt(i1))

##################################################################
################## N O R M A L I Z A C I Ó N #####################
##################################################################

mu,sigma1,M1,b0,b1 = popt
y_fit = gaussBckgrnd(xi1,mu,sigma1,M1,b0,b1)
r_fit = recta(xi1,b0,b1,mu)
plt.figure(figsize=(cm2inch(45.0),cm2inch(25.0)))
plt.plot(xi1,i1,drawstyle='steps-mid',linewidth=3,color=(0.7,0,0,0.9))
plt.plot(xi1,y_fit,lw=3)
plt.plot(xi1,r_fit)
plt.xlabel("Canal")
plt.ylabel("Cuentas/Canal")

plt.fill_between(xi1,r_fit,y_fit,color=(0,0,0.7,0.2));
co601=statsGauss_tabla(popt,pcov) # La salida produce resultados listos para incluir y editar en latex. Pero falta depurar la notación de las incertidumbres...
co601.append(round(co601[4]*2.35,2))
co601.insert(0,"Co60 Pico 1")
#########################################################################
######################## P I C O S  I N D I V I D U A L E S #############
#########################################################################

#Pico 2 Co60

#Intervalo de FORMA MANUAL para cada PICO
i1=y2[410:465]
xi1=x0[410:465]

M1=np.max(i1)

mu=xi1[0]+np.where(i1==M1)[0]
mu=float(mu)
s1=np.sqrt(mu)

popt, pcov = curve_fit(gaussBckgrnd,xi1,i1,p0=[mu,s1,M1,b0,b1])#sigma=np.sqrt(i1))

##################################################################
################## N O R M A L I Z A C I Ó N #####################
##################################################################

mu,sigma1,M1,b0,b1 = popt
y_fit = gaussBckgrnd(xi1,mu,sigma1,M1,b0,b1)
r_fit = recta(xi1,b0,b1,mu)
plt.figure(figsize=(cm2inch(45.0),cm2inch(25.0)))
plt.plot(xi1,i1,drawstyle='steps-mid',linewidth=3,color=(0.7,0,0,0.9))
plt.plot(xi1,y_fit,lw=3)
plt.plot(xi1,r_fit)
plt.xlabel("Canal")
plt.ylabel("Cuentas/Canal")

plt.fill_between(xi1,r_fit,y_fit,color=(0,0,0.7,0.2));
co602=statsGauss_tabla(popt,pcov) # La salida produce resultados listos para incluir y editar en latex. Pero falta depurar la notación de las incertidumbres...
co602.append(round(co602[4]*2.35,2))
co602.insert(0,"Co60 Pico 2")

##################################################################
################## C S 1 3 7                ######################
##################################################################

#cs137: pico 1 0-40 pico 2 200-220
#########################################################################
######################## P I C O S  I N D I V I D U A L E S #############
#########################################################################

#Pico 1 Cs137

#Intervalo de FORMA MANUAL para cada PICO
i1=y3[15:30]
xi1=x0[15:30]

M1=np.max(i1)

mu=xi1[0]+np.where(i1==M1)[0]
mu=float(mu)
s1=np.sqrt(mu)

popt, pcov = curve_fit(gaussBckgrnd,xi1,i1,p0=[mu,s1,M1,b0,b1])#sigma=np.sqrt(i1))

##################################################################
################## N O R M A L I Z A C I Ó N #####################
##################################################################

mu,sigma1,M1,b0,b1 = popt
y_fit = gaussBckgrnd(xi1,mu,sigma1,M1,b0,b1)
r_fit = recta(xi1,b0,b1,mu)
plt.figure(figsize=(cm2inch(45.0),cm2inch(25.0)))
plt.plot(xi1,i1,drawstyle='steps-mid',linewidth=3,color=(0.7,0,0,0.9))
plt.plot(xi1,y_fit,lw=3)
plt.plot(xi1,r_fit)
plt.xlabel("Canal")
plt.ylabel("Cuentas/Canal")

plt.fill_between(xi1,r_fit,y_fit,color=(0,0,0.7,0.2));
cs1371=statsGauss_tabla(popt,pcov) # La salida produce resultados listos para incluir y editar en latex. Pero falta depurar la notación de las incertidumbres...
cs1371.append(round(cs1371[4]*2.35,2))
cs1371.insert(0,"Cs137 Pico 1")
#########################################################################
######################## P I C O S  I N D I V I D U A L E S #############
#########################################################################

#Pico 2 Cs137

#Intervalo de FORMA MANUAL para cada PICO
i1=y3[200:255]
xi1=x0[200:255]

M1=np.max(i1)

mu=xi1[0]+np.where(i1==M1)[0]
mu=float(mu)
s1=np.sqrt(mu)

popt, pcov = curve_fit(gaussBckgrnd,xi1,i1,p0=[mu,s1,M1,b0,b1])#sigma=np.sqrt(i1))

##################################################################
################## N O R M A L I Z A C I Ó N #####################
##################################################################

mu,sigma1,M1,b0,b1 = popt
y_fit = gaussBckgrnd(xi1,mu,sigma1,M1,b0,b1)
r_fit = recta(xi1,b0,b1,mu)
plt.figure(figsize=(cm2inch(45.0),cm2inch(25.0)))
plt.plot(xi1,i1,drawstyle='steps-mid',linewidth=3,color=(0.7,0,0,0.9))
plt.plot(xi1,y_fit,lw=3)
plt.plot(xi1,r_fit)
plt.xlabel("Canal")
plt.ylabel("Cuentas/Canal")

plt.fill_between(xi1,r_fit,y_fit,color=(0,0,0.7,0.2));
cs1372=statsGauss_tabla(popt,pcov) # La salida produce resultados listos para incluir y editar en latex. Pero falta depurar la notación de las incertidumbres...
cs1372.append(round(cs1372[4]*2.35,2))
cs1372.insert(0,"Cs137 Pico 2")


#######################################################################
########################  C A L I B R A C I O N #######################
#######################################################################



# Imprimimos la tabla con los datos a comparar

datos=[na221,na222,cs1371,cs1372,co601,co602]   
tabla=df(datos,columns=["Dato","$\mu$","(mu)","$\sigma$","(sigma)","Intensidad","(I)","FWHM"])
print(tabla)
"""
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








