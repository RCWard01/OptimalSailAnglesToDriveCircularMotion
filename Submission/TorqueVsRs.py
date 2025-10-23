## Plot torque vs R_s for various omega
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


v_w = 1000
R_1 = 1000
rho = 1.225
A = 10000


F_vals = []
N = 12
theta = 0.1
R_vals = np.linspace(0, 1, 100)

omega_vals = np.linspace(0, 2, 5)
for omega in omega_vals:
    F_vals = []
    for R1 in R_vals:
        R = 1000 * R1
        v_s = omega * R
        S = 0
        for i in range(N):
            S += R/R_1 * F(theta + 2*np.pi*i/N, v_w, v_s, A, rho)
        S /= rho * A * v_w**2 / 2 # Dimensionless torque
        F_vals.append(S)

    plt.plot(R_vals, F_vals, label=r'$\omega R_1/V_w$ = '+str(omega*R_1/v_w))

plt.xlabel(r'$R_S/R_1$', fontsize=14)
plt.ylabel(r'Torque ($\bar{\tau}_{TOT}$)', fontsize=14)
plt.legend(fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.grid()

plt.tight_layout()
plt.show()
