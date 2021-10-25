# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 16:19:40 2021

@author: Esteban
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import random as rn

fmax = 0.86
grid = np.zeros(13)
m1 = fmax/8
m2 = -fmax/4

pdf = np.zeros(13)

for i in range(13):
    grid[i] = 0.5 + i
    if i <= 8:
        pdf[i] = m1*i
    else:
        pdf[i] = m2*i - m2*12
        
    

plt.plot(grid, pdf)
plt.show()