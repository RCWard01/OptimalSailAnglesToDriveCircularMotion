import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


df = pd.read_csv('NACA12.csv', skiprows=1, header=0)
df2 = pd.read_csv('NACA0012.csv', skiprows=1, header=0)

alpha_vals = df["Alpha"].values
cl_vals = df["Cl"].values # Re = 700000
cd_vals = df["Cd"].values
#cl2_vals = df2["Cl2"].values # Re = 360000
#cd2_vals = df2["Cd2"].values

fig1, ax1 = plt.subplots(1, 1)
fig2, ax2 = plt.subplots(1, 1)
fsize = 16

ax1.scatter(alpha_vals, cl_vals, marker='x', color='k', linewidth=0.5, label='Experimental')
ax2.scatter(alpha_vals, cd_vals, marker='x', color='k', linewidth=0.5, label='Experimental')

alpha_plot = np.linspace(0, 180)
cl_plot = np.sin(2 * alpha_plot * np.pi / 180)
cd_plot = 1.7 * np.sin(alpha_plot * np.pi / 180)**2 + 0.1

#axs[0].scatter(alpha_vals, cl2_vals, marker='o', facecolors='none', edgecolors='k', linewidth=0.5, label='$Re = 3.6 x 10^5$')
#axs[1].scatter(alpha_vals, cd2_vals, marker='o', facecolors='none', edgecolors='k', linewidth=0.5, label='$Re = 3.6 x 10^5$')

ax1.plot(alpha_plot, cl_plot, label='$\sin(2\\alpha)$')
ax2.plot(alpha_plot, cd_plot, label='$1.7 \sin^2(\\alpha) + 0.1$')

for ax in [ax1, ax2]:
    ax.grid()
    ax.legend(fontsize=fsize)
    ax.set_xlabel('Angle of Attack $\\alpha$ (radians)', fontsize=fsize+2)
    ax.set_xticks(np.array([0, 45, 90, 135, 180]), [0, '$\\pi/4$', '$\\pi/2$', '$3\\pi/4$', '$\\pi$'])
    ax.tick_params(axis='x', labelsize=fsize)
    ax.tick_params(axis='y', labelsize=fsize)

ax1.set_ylabel('Lift Coefficient $c_L$', fontsize=fsize+2)
ax2.set_ylabel('Drag Coefficient $c_D$', fontsize=fsize+2)

fig1.tight_layout()
fig2.tight_layout()

fig1.savefig('NACA12_Lift.png', dpi=1000)
fig2.savefig('NACA12_Drag.png', dpi=1000)
plt.show()