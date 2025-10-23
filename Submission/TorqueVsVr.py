## Plot torque vs V_r
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar


def theta_A(theta, v_s, v_w):
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
    a = theta_A(theta, v_r, 1)
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
    a = theta_A(theta, v_s, v_w)
    V_A = np.sqrt(v_s**2 + v_w**2 + 2*v_w*v_s*np.sin(theta))
    v_r = v_s/v_w
    return 0.5 * rho * A * V_A**2 * g(s_opt(theta, v_r), a)
    

v_w = 1
rho = 1.225
A = 10000

v_r_vals = np.linspace(0, 3, 1000)
N_vals = [4,8,12,16,20]
theta = 0.01
for N in N_vals:
    F_vals = []
    for v_s in v_r_vals:
        S = 0
        for i in range(N):
            S += F(theta + 2*np.pi*i/N, v_w, v_s, A, rho)
        S /= rho * A * v_w**2 / 2 # Dimensionless torque
        F_vals.append(S)

    plt.plot(v_r_vals, F_vals, label=r'$N$ = '+str(N))

fsize = 16
plt.xlabel(r'$V_r$', fontsize=fsize)
plt.ylabel(r'Torque ($\bar{\tau}_{TOT}$)', fontsize=fsize)
plt.legend(fontsize=fsize)
plt.xticks(fontsize=fsize)
plt.yticks(fontsize=fsize)
plt.grid()

plt.tight_layout()
plt.show()