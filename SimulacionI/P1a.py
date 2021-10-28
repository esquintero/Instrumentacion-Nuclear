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
       


ax=plt.gca()
plt.plot(grid, pdf, 'bo')
plt.grid(True)

plt.figure()
plt.plot(grid, F, ds='steps-mid',c='r')
ax=plt.gca()
ax.set_xticks(range(0,12))
plt.grid(True)


plt.show()


