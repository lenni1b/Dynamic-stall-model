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


class f_param:
    linslope = 0
    popt = []
    stall = 0


def heavyside(x):
    """
    unit step function, returns 1 when positive, and 0 when negative
    """
    return 0.5 * (np.sign(x) + 1)


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


def separationpoint(angle):
    stall = f_param.stall
    transition = 1.1*np.pi/180
    if angle > stall + transition:
        output = fourier12(angle, *f_param.popt)
    elif angle > stall:
        output = 1-((1 - fourier12((stall+transition), *f_param.popt)) /
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

    f = np.empty([len(cl0)])
    # Find separation point function
    for n in range(len(cl0)):
        if abs(cl0[n]) < 0.0001:
            inside = 0
            theta = 1/0.25
        else:
            inside = - (cl[n] / cl0[n])**(0.25)
            theta = np.arccos(inside) / (0.25)

        f[n] = 0.5 * (1 + np.cos(theta))

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

    f_param.linslope = dalpha0
    f_param.popt = popt_f
    f_param.stall = stallangle
    # Plot the fitted function for f
    plot_angles = np.arange(0, 55, 0.05)*np.pi/180
    plot_f = []
    for angle in plot_angles:
        plot_f.append(fourier12(angle, *popt_f))
    plt.plot(plot_angles*180/np.pi, plot_f)

    plot_f_final = []
    for angle in plot_angles:
        plot_f_final.append(separationpoint(abs(angle)))

    plt.plot(plot_angles*180/np.pi, plot_f_final)
    plt.ylim([0, 1.1])
    plt.xlim([0, 55])

    return f_param


def separation_angle(alpha):
    """
    Finds the separation angle, when the foil is mapped on a unit circle
    """
    f = separationpoint(alpha)
    return np.arccos(2 * f - 1)

def larsen_model(alpha, time):
    """
    Returns Cl and CM for the given angle of attack, based on the larsen
    dynamic stall model
    """
    u_inf = 0.525
    chord = 0.075
    # Dynamic parameters (Vertol 23010-1.58-profile)
    OMEGA1 = 0.0455 * (2*u_inf / chord)
    OMEGA2 = 0.3 * (2*u_inf / chord)
    OMEGA3 = 0.1 * (2*u_inf / chord)
    OMEGA4 = 0.075 * (2*u_inf / chord)
    A1 = 0.165
    A2 = 0.335
    ALPHAv = 13*np.pi/180

    # State variable matrix
    z = np.empty([len(alpha), 4]) * 0
    cl_dyn = np.empty([len(alpha)]) * 0
    dt = time[1]-time[0]
    for n in range(len(time)-1):
        # Find change in linear static lift
        d_cl0 = f_param.linslope * (alpha[n+1] - alpha[n]) / dt
        # Solve the first 3 state variables
        z[n+1, 0] = z[n, 0] + dt *(- OMEGA1 * z[n, 0] + A1 * d_cl0)
        z[n+1, 1] = z[n, 1] + dt *(- OMEGA2 * z[n, 1] + A2 * d_cl0)
        z[n+1, 2] = z[n, 2] + dt *(- OMEGA3 * z[n, 2] + OMEGA3 *
                                   separation_angle(alpha[n]))
        # Find change in additional dynamic lift
        d_dyn_lift = (dyn_lift(alpha[n+1], z[n+1, 0], z[n+1, 1], z[n+1, 0]) -
                      dyn_lift(alpha[n], z[n, 0], z[n, 1], z[n, 0])) / dt
        # find change in angle of attack
        d_alpha = (alpha[n+1] - alpha[n]) / dt
        # find fourth state variable, induced lift due to pressure and vortex
        z[n+1, 3] = z[n, 3] + dt *(- OMEGA4 * z[n, 3] +
                                   d_dyn_lift *
                                   heavyside(ALPHAv - alpha[n]) *
                                   heavyside(d_alpha))
        # Calculate dynamic lift from state variables
        cl_dyn[n+1] = (np.cos(0.25 * z[n+1, 2])**4 *
                       (f_param.linslope * alpha[n+1] - z[n+1, 0] - z[n+1, 1]) +
                      z[n+1, 3])
    return cl_dyn, z



def lin_lift(alpha):
    """"
    Find the linear static lift
    """
    f = separationpoint(alpha)
    if f > 0:
        lift = f_param.linslope * alpha
    elif f <= 0:
        raise Exception("Linear lift is not implemented yet for f = 0")
    return lift


def dyn_lift(alpha, c1, c2, theta_d):
    """
    Calculates the dynamic lift contribution
    """
    # Static linear lift
    cl0 = f_param.linslope * alpha
    # linear dynamic lift for attached flow
    cl0d = cl0 - c1 - c2
    # reduction in dynamic linear lift
    cld = np.cos(0.25 * theta_d)**4 * cl0d
    return cl0d - cld

