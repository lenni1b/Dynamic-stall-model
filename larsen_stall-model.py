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
plt.close("all")
def force_to_coefficient(force, u_inf):
    """
    Converts the input force to a lift or drag coefficient, based on geometries
    in the CWT experiments
    """
    CHORD = 0.075
    RHO = 998.2  # Water at 20 degrees celsius
    return 2 * force / (RHO * CHORD * u_inf**2)

# fourier series defintions
tau = 0.97715
def fourier12(x, C, tau, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12):
    return C + a1 * np.cos(1 * np.pi / tau * x) + \
           a2 * np.cos(2 * np.pi / tau * x) + \
           a3 * np.cos(3 * np.pi / tau * x) + \
           a4 * np.cos(4 * np.pi / tau * x) + \
           a5 * np.cos(5 * np.pi / tau * x) + \
           a6 * np.cos(6 * np.pi / tau * x) + \
           a7 * np.cos(7 * np.pi / tau * x) + \
           a8 * np.cos(7 * np.pi / tau * x) + \
           a9 * np.cos(7 * np.pi / tau * x) + \
           a10 * np.cos(7 * np.pi / tau * x) + \
           a11 * np.cos(7 * np.pi / tau * x) + \
           a12 * np.cos(8 * np.pi / tau * x)
# static stall approximation
stallangle = 5.8 * np.pi/180
# Import measurements
angles = []
cl = []
mz = []
with open('static_results_zigzag.csv', 'r') as f:
    reader = csv.DictReader(f, delimiter=';')
    for row in reader:
        cl.append(float(row['cl']))
        mz.append(float(row['mz']))
        angles.append(float(row['angle']) * np.pi/180)

plt.plot(angles, cl)

# Fit a static lift coefficient slope alpha zero
i = 0
cl_prestall = []
angles_prestall = []
for angle in angles:
    if np.abs(angle) < stallangle:
        cl_prestall.append(cl[i])
        angles_prestall.append(angle)
    i += 1
def linear(x, C, offset):  # linear function
    return np.multiply(C, x) + offset
popt, pcov =curve_fit(linear, angles_prestall, cl_prestall)
plt.plot([-stallangle, stallangle], linear([-stallangle, stallangle], popt[0], 0),
         c='g', linestyle='--')
dalpha0 = popt[0]  # static linear lift coefficient
cl0 = np.multiply(dalpha0, angles)

# Find separation point function
inside = np.power(np.divide(cl, cl0), (1.0/4.0))
theta = 4 * np.arccos(inside)
f = 0.5 * (1 + np.cos(theta))
del inside

# Remove f values for angles below stall angle
f_temp = []
angles_temp = []
for i in range(len(f)):
    if abs(angles[i]) > stallangle:
        if np.isnan(f[i]) == 0:
            angles_temp.append(abs(angles[i]))
            f_temp.append(f[i])

plt.figure(2)
plt.scatter(np.abs(angles_temp)*180/np.pi, f_temp, s=10)
# Fit a function to the separation point function f

popt1, pcov = curve_fit(fourier12, angles_temp, f_temp, maxfev=5000)
plot_angles = np.arange(0, 20, 0.05)*np.pi/180
plot_f = []
for angle in plot_angles:
    plot_f.append(fourier12(angle, *popt1))
plt.plot(plot_angles*180/np.pi, plot_f)

# add a straight line, and fit a logarithmic transition
def separationpoint(angle):
    transition = 0.5*np.pi/180
    if angle > stallangle + transition:
        output = fourier12(angle, *popt1)
    elif angle > stallangle:
        output = 1-((1 - fourier12((stallangle+transition), *popt1)) /
                  transition) * (angle-stallangle)
    else:
        output = 1
    return output

plot_f_final = []
for angle in plot_angles:
    plot_f_final.append(separationpoint(abs(angle)))

plt.plot(plot_angles*180/np.pi, plot_f_final)
plt.ylim([0, 1.1])
plt.xlim([0, 25])
