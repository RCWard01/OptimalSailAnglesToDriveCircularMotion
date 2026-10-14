#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 30 15:34:53 2024

@author: roryward
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar


def alpha(theta, v_s, v_w):
    num = v_s + v_w * np.sin(theta)
    den = np.sqrt(v_s**2 + v_w**2 + 2*v_s*v_w*np.sin(theta))
    if den == 0:
        return 0
    if abs(num) > abs(den):
        return np.sign(np.cos(theta))*np.arccos(np.sign(num))
    return np.sign(np.cos(theta))*np.arccos(num/den)
        

def c_L(x):
    return np.sin(2*x)

def c_D(x):
    return (1.7/2) * (1 - np.cos(2*x))+0.1

def g(s, a):
    if a > 0:
        if  s > a or s < a - np.pi/2:
            return 0
    elif a < 0:
        if s < a + np.pi or s > np.pi*3/2 + a:
            return 0
    return c_L(a - s) * np.sin(a) - c_D(a - s) * np.cos(a)


def s_opt(theta, v_r):
    a = alpha(theta, v_r, 1)
    if a > 0:
        xl = a - np.pi/2
        xr = a
    else:
        xl = a + np.pi
        xr = np.pi*3/2 + a
    
    def f(s):
        return - g(s, a)
    
    sol = minimize_scalar(f, bounds = [xl, xr])
    return sol.x

def F(theta, v_w, v_s, A, rho):
    a = alpha(theta, v_s, v_w)
    V_A = np.sqrt(v_s**2 + v_w**2 + 2*v_w*v_s*np.sin(theta))
    v_r = v_s/v_w
    return V_A**2 * g(s_opt(theta, v_r), a)
    

v_w = 1
rho = 1.225  # kg/m^3
A = 10000  # m^2, seems large?

v_r_vals = np.linspace(0, 3, 1000)
N_vals = [100]
theta = 0.01
first = True
for N in N_vals:
    F_vals = []
    for v_s in v_r_vals:
        S = 0
        for i in range(N):
            S += F(theta + 2*np.pi*i/N, v_w, v_s, A, rho)
        S /= v_w**2
        if S < 0 and first:
            print("First negative torque at v_r = ", v_s)
            first = False
        F_vals.append(S)

    plt.plot(v_r_vals, np.array(F_vals) / N, label=r'$N$ = '+str(N))

fsize = 16
plt.xlabel(r'$V_r$', fontsize=fsize)
plt.ylabel(r'Torque ($\bar{\tau}_{TOT} / N$)', fontsize=fsize)
#plt.legend(fontsize=fsize)
plt.xticks(fontsize=fsize)
plt.yticks(fontsize=fsize)
#plt.title(r'$N = $'+str(N))
plt.grid()

plt.hlines(0, color="k", lw=1, xmin=-0.15, xmax=3.15)
plt.vlines(1.545, color="k", lw=1, ls="--", ymin=min(F_vals) / N - 0.1, ymax=0)
plt.xlim(-0.15, 3.15)
plt.ylim(min(F_vals) / N - 0.1, 1)
plt.text(1.57, -0.725, r"$V_{r0}$", fontsize=fsize)
plt.tight_layout()
fig = plt.gcf()
#fig.savefig('TorqueVsVrScaled.png', dpi=500)

plt.show()