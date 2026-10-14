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

def c_D(x, d):
    return (1.7/2) * (1 - np.cos(2*x)) + d

def g(s, a, d):
    if a > 0:
        if  s > a or s < a - np.pi/2:
            return 0
    elif a < 0:
        if s < a + np.pi or s > np.pi*3/2 + a:
            return 0
    return c_L(a - s) * np.sin(a) - c_D(a - s, d) * np.cos(a)


def s_opt(theta, v_r, d):
    a = alpha(theta, v_r, 1)
    if a > 0:
        xl = a - np.pi/2
        xr = a
    else:
        xl = a + np.pi
        xr = np.pi*3/2 + a
    
    def f(s):
        return - g(s, a, d)
    
    sol = minimize_scalar(f, bounds = [xl, xr])
    return sol.x

def F(theta, v_w, v_s, d):
    a = alpha(theta, v_s, v_w)
    V_A = np.sqrt(v_s**2 + v_w**2 + 2*v_w*v_s*np.sin(theta))
    v_r = v_s/v_w
    return V_A**2 * g(s_opt(theta, v_r, d), a, d)
    

v_w = 1

v_r_vals = np.linspace(0, 3, 1000)
N = 50
theta = 0.01
d_vals = np.logspace(-2, 1, 20)
vr0_vals = []
def compute_S(d, v_s, N, theta, v_w):
    S = 0.0
    for i in range(N):
        S += F(theta + 2*np.pi*i/N, v_w, v_s, d)
    return S / (v_w**2)


vr0_vals = []

tol = 1e-4
max_iter = 50

for d in d_vals:

    # --- Step 1: bracket the root ---
    v_low = 0.0
    v_high = 1.0

    S_low = compute_S(d, v_low, N, theta, v_w)
    S_high = compute_S(d, v_high, N, theta, v_w)

    # expand bracket until sign change
    while S_low > 0 and S_high > 0:
        v_high *= 2
        S_high = compute_S(d, v_high, N, theta, v_w)

    # --- Step 2: bisection ---
    for _ in range(max_iter):
        v_mid = 0.5 * (v_low + v_high)
        S_mid = compute_S(d, v_mid, N, theta, v_w)

        if abs(S_mid) < tol:
            break

        if S_mid > 0:
            v_low = v_mid
            S_low = S_mid
        else:
            v_high = v_mid

    vr0_vals.append(v_mid)

    

plt.semilogx(d_vals, vr0_vals)

fsize = 16
plt.xlabel(r'$d$', fontsize=fsize)
plt.ylabel(r'$V_{r0}$', fontsize=fsize)
#plt.legend(fontsize=fsize)
plt.xticks(fontsize=fsize)
plt.yticks(fontsize=fsize)
#plt.title(r'$N = $'+str(N))
plt.grid()

plt.tight_layout()
fig = plt.gcf()
fig.savefig('Vr0Vsd.png', dpi=500)

plt.show()