## Plot the apparent wind angle theta_A vs theta
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

def theta_A(theta, v_s, v_w):
    num = v_s + v_w * np.cos(theta)
    den = np.sqrt(v_s**2 + v_w**2 + 2*v_s*v_w*np.cos(theta))
    if den == 0:
        return 0
    if abs(num) > abs(den):
        return -np.sign(np.sin(theta))*np.arccos(np.sign(num))
    return np.pi  * (np.sign(np.sin(theta)) + 1) - np.sign(np.sin(theta))*np.arccos(num/den)


theta_store = np.linspace(0, 2*np.pi, 1000)

colours = ['k', 'b', 'r', 'g']
i = 0
for v_r in [0.1, 0.9, 1.1, 10]:
    c = colours[i]
    alpha_store = []
    for theta in theta_store:
        alpha_store.append(theta_A(theta, v_r, 1))

    if i in [0, 1]:  # black and blue
        alpha_store2 = []
        for a in alpha_store[250:750]:
            alpha_store2.append(2*np.pi + a)

    plt.plot(theta_store[1:500], alpha_store[1:500], color=c)
    plt.plot(theta_store[500:], alpha_store[500:],
             label=r'$V_r$ =' + str(v_r), color=c)
    i += 1

matplotlib.rc('xtick', labelsize=40)
plt.legend(fontsize=14)
plt.xlabel(r'$\theta$', fontsize=14)
plt.ylabel(r'$\theta_A$', fontsize=14)
plt.xticks([0, np.pi/2, np.pi, np.pi*3/2, 2*np.pi],
           [0, r'$\pi/2$', r'$\pi$', r'$3\pi/2$', r'$2\pi$'], fontsize=14)
plt.yticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi],
           [0, r'$\pi/2$',
           r'$\pi$', r'$3\pi/2$',
           r'$2\pi$'], fontsize=14)
plt.title('Apparent Wind Angle', fontsize=18)
plt.grid()

matplotlib.rc('xtick', labelsize=100)
plt.show()
