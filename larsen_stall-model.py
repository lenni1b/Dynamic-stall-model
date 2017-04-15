# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 13:44:07 2017
Dynamic stall model for wind turbine airfoils, method by J.W.Larsen (2007)
@author: Lennard
"""

import csv
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from dynstall_submodule import find_ffunction_parameters, separationpoint, heavyside
import numpy as np
plt.figure(2)
plt.clf()
plt.figure(1)
plt.clf()


# static stall approximation
stallangle = 5* np.pi/180

# Import measurements
angles = []
cl = []
mz = []
csvfiles = ['static_results_zigzag_deepstall.csv',
            'static_results_zigzag.csv']
for n in range(2):
    with open(csvfiles[n], 'r') as f:
        reader = csv.DictReader(f, delimiter=';')
        for row in reader:
            cl.append(float(row['cl']))
            mz.append(float(row['mz']))
            angles.append(float(row['angle']) * np.pi/180)
plt.figure(1)
plt.plot(np.multiply(angles, 180/np.pi), cl)

# Determine the parameters for a function of the separationpoint vs angle
f_param = find_ffunction_parameters(stallangle, cl, angles)




