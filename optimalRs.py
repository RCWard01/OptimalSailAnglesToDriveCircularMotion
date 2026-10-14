#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar


# ----------------------------
# Aerodynamic / model functions
# ----------------------------

def alpha(theta, v_s, v_w):
    num = v_s + v_w * np.sin(theta)
    den = np.sqrt(v_s**2 + v_w**2 + 2 * v_s * v_w * np.sin(theta))

    if den == 0:
        return 0.0

    ratio = num / den
    ratio = np.clip(ratio, -1.0, 1.0)

    return np.sign(np.cos(theta)) * np.arccos(ratio)


def c_L(x):
    return np.sin(2 * x)


def c_D(x):
    return (1.7 / 2) * (1 - np.cos(2 * x)) + 0.1


def g(s, a):
    # keep your domain restrictions
    if a > 0:
        if s > a or s < a - np.pi / 2:
            return 0.0
    elif a < 0:
        if s < a + np.pi or s > 1.5 * np.pi + a:
            return 0.0

    return c_L(a - s) * np.sin(a) - c_D(a - s) * np.cos(a)


# ----------------------------
# Optimisation over s
# ----------------------------

def s_opt(theta, v_s, v_w):
    a = alpha(theta, v_s, v_w)

    if a > 0:
        bounds = (a - np.pi / 2, a)
    else:
        bounds = (a + np.pi, 1.5 * np.pi + a)

    def objective(s):
        return -g(s, a)

    res = minimize_scalar(objective, bounds=bounds, method="bounded")
    return res.x


# ----------------------------
# Main force function
# ----------------------------

def F(theta, v_w, v_s):
    a = alpha(theta, v_s, v_w)
    V_A = np.sqrt(v_s**2 + v_w**2 + 2 * v_w * v_s * np.sin(theta))

    s_star = s_opt(theta, v_s, v_w)

    return V_A**2 * g(s_star, a)


# ----------------------------
# Torque / objective over R
# ----------------------------

def torque(R, omega, theta_grid, v_w, R1):
    v_s = omega * R

    S = 0.0
    for theta in theta_grid:
        S += (R / R1) * F(theta, v_w, v_s)

    return S / v_w**2


def objective_R(R, omega, theta_grid, v_w, R1):
    return -torque(R, omega, theta_grid, v_w, R1)


# ----------------------------
# Run optimisation sweep
# ----------------------------

v_w = 1
R_1 = 10

N = 50
theta0 = 0.1

theta_grid = theta0 + 2 * np.pi * np.arange(N) / N

omegaR1_vals = np.logspace(-2, 2, 10)
optimalRs_vals = []

bounds = (0.0, R_1)

for omegaR1 in omegaR1_vals:
    omega = omegaR1 / R_1

    res = minimize_scalar(
        objective_R,
        bounds=bounds,
        method="bounded",
        args=(omega, theta_grid, v_w, R_1)
    )

    optimalRs_vals.append(res.x / R_1)


# ----------------------------
# Plot
# ----------------------------

plt.loglog(omegaR1_vals, optimalRs_vals)
linear_plot = np.linspace(1, 100, 5)
plt.plot(linear_plot, 1/linear_plot, 'k--', label=r'1 / $\omega R_1/V_w$')

plt.xlabel(r'$\omega R_1/V_w$', fontsize=14)
plt.ylabel(r'Optimal $R_S/R_1$', fontsize=14)

plt.grid()
plt.tight_layout()
fig = plt.gcf()
#fig.savefig('OptimalRsLOGLOG.png', dpi=500)
plt.show()

print(omegaR1_vals * np.array(optimalRs_vals))