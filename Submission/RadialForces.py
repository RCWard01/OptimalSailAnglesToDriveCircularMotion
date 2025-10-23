## Plot radial force vs theta for various v_r
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar

def theta_A(theta, v_s, v_w):
    num = v_s + v_w * np.cos(theta)
    den = np.sqrt(v_s**2 + v_w**2 + 2*v_s*v_w*np.cos(theta))
    if den == 0:
        return 0
    if abs(num) > abs(den):
        return -np.sign(np.sin(theta))*np.arccos(np.sign(num))
    return -np.sign(np.sin(theta))*np.arccos(num/den)

def c_L(x): # Lift coefficient
    return np.sin(2*x)

def c_D(x): # Drag coefficient
    return (1.7/2) * (1 - np.cos(2*x)) + 0.1

def g(s, a):
    if a > 0:
        if  s > a or s < a - np.pi/2:
            print('here')
            return 0
    elif a < 0:
        if s < a + np.pi or s > np.pi*3/2 + a:
            print('here2')
            return 0
    return c_L(a - s) * np.sin(a) - c_D(a - s) * np.cos(a)


colours = ['k', 'b', 'r', 'k', 'b', 'r']
i = 0
fsize = 16
fig1, ax1 = plt.subplots(1, 1)
fig2, ax2 = plt.subplots(1, 1)
for v_s in [0, 0.5, 1, 1.5]:
    theta_vals = np.linspace(0, 2*np.pi, 1000)
    s_vals = []
    a_vals = []
    v_w = 1
    v_r = v_s/v_w
    for theta in theta_vals:
        a = theta_A(theta, v_s, v_w)
        if a <= 0:
            a_vals.append(a + 2*np.pi)
        else:
            a_vals.append(a)
        if a > 0:
            xl = a - np.pi/2
            xr = a
        else:
            xl = a + np.pi
            xr = np.pi*3/2 + a
        
        def f(s):
            return - g(s, a)
        sol = minimize_scalar(f, bounds = [xl, xr])
        s_vals.append(sol.x)
    
    s_vals = np.array(s_vals)
    a_vals = np.array(a_vals)
    colour = colours[i]
    i += 1


    F_vals = []
    F1_vals = []
    for j in range(len(s_vals)):
        theta, s, a = theta_vals[j], s_vals[j], a_vals[j]
        F = (v_r ** 2 + 1 + 2 * v_r * np.cos(theta)) * (c_L(a-s)*np.cos(a) + c_D(a-s) * np.sin(a))
        F_vals.append(F)
        F1_vals.append(F * np.sin(theta))
    
    ax1.plot(theta_vals, F_vals, label=r'$V_r = $' + str(v_r))
    ax1.set_ylabel(r'$\hat{F}_{RAD}$', fontsize=fsize)
    ax1.set_xlabel(r'$\theta$', fontsize=fsize)
    ax1.set_xticks([0, np.pi/2, np.pi, np.pi*3/2, 2*np.pi], [0, r'$\pi/2$', r'$\pi$', r'$3\pi/2$', r'$2\pi$'], fontsize=fsize)
    ax1.tick_params(axis='y', labelsize=fsize)
    ax1.grid(True)
    
    ax2.plot(theta_vals, F1_vals, label=r'$V_r = $' + str(v_r))
    ax2.set_ylabel(r'$\hat{F}_{RAD} \cdot \sin \theta$', fontsize=fsize)
    ax2.set_xlabel(r'$\theta$', fontsize=fsize)
    ax2.set_xticks([0, np.pi/2, np.pi, np.pi*3/2, 2*np.pi], [0, r'$\pi/2$', r'$\pi$', r'$3\pi/2$', r'$2\pi$'], fontsize=fsize)
    ax2.tick_params(axis='y', labelsize=fsize)
    ax2.grid(True)
    ax2.legend(fontsize=fsize)
    plt.ylim(-2.5, 2.5)


fig1.tight_layout()
fig2.tight_layout()
plt.show()
