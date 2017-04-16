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
from dynstall_submodule import find_ffunction_parameters, larsen_model
from dynstall_submodule import lin_lift
import numpy as np
plt.figure(2)
plt.clf()
plt.figure(1)
plt.clf()
plt.close("all")

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
f_param = find_ffunction_parameters(stallangle, np.abs(cl), np.abs(angles))

# positions to be simulated
dt = 0.005 # timestep
t_total = 5 # time to reach max angle
time = np.arange(0, t_total, dt)
alpha = time * (1/t_total) * 30 * np.pi / 180


cl_dyn, z = larsen_model(alpha, time)

plt.figure(3)
plt.scatter(np.abs(angles), np.abs(cl))
plt.plot(alpha, cl_dyn)

plt.figure(4)
# create a plot with lift components
cl0d = np.empty(len(cl_dyn))
cld = np.empty(len(cl_dyn))
for n in range(len(cl_dyn)):
    cl0d[n] = lin_lift(alpha[n]) - z[n, 0] - z[n, 1]
    cld[n] = np.cos(0.25*z[n, 2])*cl0d[n]
plt.plot(alpha, cl0d, label="cl0d")
plt.plot(alpha, cld, label="cld")
plt.plot(alpha, z[:,3], label="clv")
plt.legend(loc="best")





