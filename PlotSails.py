## Plot optimal sail orientations on a circle
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
from matplotlib.patches import Arc, FancyArrowPatch


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
    return  (1.7/2) * (1 - np.cos(2*x)) + 0.1

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

fig = plt.figure()
ax = fig.add_subplot()
ax.set_aspect('equal', adjustable='box')

circle_x = np.linspace(-1, 1, 1000)
circle_y1 = np.sqrt(1 - circle_x**2)
circle_y2 = - np.sqrt(1 - circle_x[::-1]**2)
plt.plot(circle_x, circle_y1, 'k--', linewidth=0.5)
plt.plot(circle_x[::-1], circle_y2, 'k--', linewidth=.5)


v_r = 0.9 # Set desired velocity ratio v_r
L = 1/5
N = 40

theta_mults = np.linspace(0, 12, N)
for mult in theta_mults:
    theta = mult * (np.pi / 6) + 0.1
    s = s_opt(theta, v_r)
    x0 = np.sin(theta)
    y0 = - np.cos(theta)
    dx = L * np.cos(theta + s)
    dy = L * np.sin(theta + s)
    plt.arrow(x0-0.5*dx, y0-0.5*dy, dx, dy, color='r', head_width=0)

plt.arrow(0,0.25,0,-0.5, head_width=0.1)
plt.text(0.1, -0.1, 'Wind', fontsize=22)
plt.axis('off')
plt.title(r'$V_r = $'+str(v_r), fontsize=16)

## Plot circular arrows
def add_circular_arrow(ax, center, radius, theta1, theta2, color='k', lw=2, arrow_length=0.15, head_stick_out=0.05):
    arc = Arc(center, 2*radius, 2*radius, angle=0,
              theta1=theta1, theta2=theta2,
              linewidth=lw, color=color)
    ax.add_patch(arc)

    end_angle = np.deg2rad(theta2)
    x_end = center[0] + radius * np.cos(end_angle)
    y_end = center[1] + radius * np.sin(end_angle)

    tangent_dir = np.array([-np.sin(end_angle), np.cos(end_angle)])

    arrow_end = np.array([x_end, y_end]) + head_stick_out * tangent_dir
    arrow_start = arrow_end - arrow_length * tangent_dir

    # Add arrowhead
    arrow = FancyArrowPatch(arrow_start, arrow_end,
                            arrowstyle='-|>,head_length=6,head_width=5',
                            color=color, linewidth=lw)
    ax.add_patch(arrow)


add_circular_arrow(ax, center=(0, 0), radius=1.2, theta1=0, theta2=45, color='k')
add_circular_arrow(ax, center=(0, 0), radius=1.2, theta1=180, theta2=225, color='k')

ax.set_aspect('equal')
ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-1.5, 1.5)
ax.axis('off')

plt.show()
