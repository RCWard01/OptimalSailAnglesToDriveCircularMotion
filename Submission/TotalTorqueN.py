## Plot total torque from N sails
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar

N = 4

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
    return (1.7/2) * (1 - np.cos(2*x)) + 0.1

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
    

v_w = 1  # m/s
v_s = 0  # m/s
rho = 1.225  # kg/m^3
A = 10000  # m^2, seems large?

fig, ax = plt.subplots()

theta_vals = np.linspace(0, 2*np.pi, 1000)
F_vals = []
fsize = 16
for v_s in [0, 0.5, 1, 2]:
    F_vals = []
    for theta in theta_vals:
        S = 0
        for i in range(N):
            S += F(theta + 2*np.pi*i/N, v_w, v_s, A, rho)
        S /= rho * A * v_w**2 / 2 # Dimensionless torque
        F_vals.append(S)
    
    # Shift theta by pi/2 after changing of theta definition (to match paper)
    F_plot = F_vals[251:] + F_vals[:251]
    plt.plot(theta_vals, F_plot, label=r'$V_r = $'+str(v_s), linewidth=2)

plt.xlabel(r'$\theta$', fontsize=fsize)
plt.ylabel(r'Torque ($\tau^*$)', fontsize=fsize)
plt.xticks([ 0, np.pi/2, np.pi, np.pi*3/2, 2*np.pi], [0, r'$\pi/2$', r'$\pi$', r'$3\pi/2$', r'$2\pi$'], fontsize=fsize)
plt.yticks(fontsize=fsize) 
plt.legend(fontsize=fsize)
plt.title(r'$N = $'+str(N), fontsize=18)
plt.grid()

plt.tight_layout()
plt.show()
