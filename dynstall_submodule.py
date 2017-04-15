# -*- coding: utf-8 -*-
"""
Created on Sat Apr 15 17:03:06 2017

@author: pivuser
"""

import csv
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

plt.figure(2)
plt.clf()
plt.figure(1)
plt.clf()

def heavyside(x):
    """
    unit step function, returns 1 when positive, and 0 when negative
    """
    return 0.5 * (numpy.sign(x) + 1)


def force_to_coefficient(force, u_inf):
    """
    Converts the input force to a lift or drag coefficient, based on geometries
    in the CWT experiments
    """
    CHORD = 0.075
    RHO = 998.2  # Water at 20 degrees celsius
    return 2 * force / (RHO * CHORD * u_inf**2)


def linear(x, C, offset):  # linear function
    return np.multiply(C, x) + offset


def fourier12(x, C, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12):
    # fourier series defintions
    tau = 0.7
    return C + a1 * np.cos(1 * np.pi / tau * x) + \
           a2 * np.cos(2 * np.pi / tau * x) + \
           a3 * np.cos(3 * np.pi / tau * x) + \
           a4 * np.cos(4 * np.pi / tau * x) + \
           a5 * np.cos(5 * np.pi / tau * x) + \
           a6 * np.cos(6 * np.pi / tau * x) + \
           a7 * np.cos(7 * np.pi / tau * x) + \
           a8 * np.cos(8 * np.pi / tau * x) + \
           a9 * np.cos(9 * np.pi / tau * x) + \
           a10 * np.cos(10 * np.pi / tau * x) + \
           a11 * np.cos(11 * np.pi / tau * x) + \
           a12 * np.cos(12 * np.pi / tau * x)


def separationpoint(angle, f_param):
    stall = f_param["stall"]
    transition = 1.1*np.pi/180
    if angle > stall + transition:
        output = fourier12(angle, *f_param["fourier"])
    elif angle > stall:
        output = 1-((1 - fourier12((stall+transition), *f_param["fourier"])) /
                  transition) * (angle-stall)
    else:
        output = 1
    return output



def find_ffunction_parameters(stallangle, cl, angles):


    # Fit a static lift coefficient slope alpha zero
    i = 0
    cl_prestall = []
    angles_prestall = []
    for angle in angles:
        if np.abs(angle) < stallangle:
            cl_prestall.append(cl[i])
            angles_prestall.append(angle)
        i += 1

    # Fit a linear transition from fourier function to constant
    popt, pcov =curve_fit(linear, angles_prestall, cl_prestall)
    plt.plot(np.multiply([-stallangle, stallangle], 180/np.pi), linear([-stallangle, stallangle], popt[0], 0),
             c='g', linestyle='--')
    dalpha0 = popt[0]  # static linear lift coefficient
    cl0 = np.multiply(dalpha0, angles)

    # Find separation point function
    inside = - np.power(np.divide(cl, cl0), (1.0/4.0))
    theta = np.arccos(inside) / (1/4)
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

    # Plot f datapoints
    plt.figure(2)
    plt.scatter(np.abs(angles_temp)*180/np.pi, f_temp, s=10)

    # Fit a function to the separation point function f
    popt_f, pcov = curve_fit(fourier12, angles_temp, f_temp, maxfev=50000)

    # Return fit parameters as a dict
    f_param = {"stall" : stallangle,
                    "fourier" : popt_f,
                    "linslope" : dalpha0}

    # Plot the fitted function for f
    plot_angles = np.arange(0, 55, 0.05)*np.pi/180
    plot_f = []
    for angle in plot_angles:
        plot_f.append(fourier12(angle, *popt_f))
    plt.plot(plot_angles*180/np.pi, plot_f)

    plot_f_final = []
    for angle in plot_angles:
        plot_f_final.append(separationpoint(abs(angle), f_param))

    plt.plot(plot_angles*180/np.pi, plot_f_final)
    plt.ylim([0, 1.1])
    plt.xlim([0, 55])

    return f_param


def separation_angle(alpha, f_param):
    """
    Finds the separation angle, when the foil is mapped on a unit circle
    """
    f = separationpoint(alpha, f_param)
    return np.arccos(2 * f - 1)

def larsen_model(alpha, f_param):
    """
    Returns Cl and CM for the given angle of attack, based on the larsen
    dynamic stall model
    """

    # Dynamic parameters (Vertol 23010-1.58-profile)
    OMEGA1 = 0.0455
    OMEGA2 = 0.3
    OMEGA3 = 0.1
    OMEGA4 = 0.075
    A1 = 0.165
    A2 = 0.335
    ALFAv = 14.75

    # Built state variable model
    A = np.matrix([-OMEGA1, 0, 0, 0],
                  [0, -OMEGA2, 0, 0],
                  [0, 0, -OMEGA3, 0],
                  [0, 0, 0, -OMEGA4])

    b0 = np.matrix([0],
                   [0],
                   [OMEGA3 * separation_angle(alpha, f_param)],
                   [0])

    b1 = np.matrix([A1],
                   [A2],
                   [0],
                   [0])
