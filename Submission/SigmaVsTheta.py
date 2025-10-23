## Plot sigma vs theta
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

def c_L(x):
    return np.sin(2*x)

def c_D(x):
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



colours = ['k', 'b', 'r']
i = 0
fsize = 16
fig, ax = plt.subplots(1, 1)
theta_vals = np.linspace(0, 2*np.pi, 1000)
for v_r in [0, 0.5, 0.9]: # Choose v_r values
    s_vals = []
    a_vals = []
    for theta in theta_vals:
        a = theta_A(theta, v_r, 1)
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

    ax.plot(theta_vals[1:500], s_vals[1:500], color=colour, label=r'$V_r$ = '+str(v_r))
    ax.plot(theta_vals[500:-1], s_vals[500:-1], color=colour)
    ax.plot(theta_vals[1:500], a_vals[1:500]/2, '--', color=colour) # Plot theta_A / 2 as dashed line
    ax.plot(theta_vals[500:-1], a_vals[500:-1]/2, '--', color=colour)


ax.legend(fontsize=fsize)
ax.set_xlabel('$\\theta$', fontsize=fsize)
ax.set_ylabel('$\\sigma^*$', fontsize=fsize)
ax.set_xticks([0, np.pi/2, np.pi, np.pi*3/2, 2*np.pi], [0, r'$\pi/2$', r'$\pi$', r'$3\pi/2$', r'$2\pi$'], fontsize=fsize)
ax.set_yticks([0, np.pi/4, np.pi/2, np.pi*3/4, np.pi], [0, r'$\pi/4$', r'$\pi/2$', r'$3\pi/4$', r'$\pi$'], fontsize=fsize)
ax.grid()

fig.tight_layout()
plt.show()


