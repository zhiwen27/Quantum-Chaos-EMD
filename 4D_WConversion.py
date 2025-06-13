
import math
import numpy as np
import pandas as pd
from scipy.integrate import simpson as simps
from joblib import Parallel, delayed, parallel_backend
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
start_time = time.time()


print("Here I start again!")

m=1
w=1
h_bar=1


x1_min, x1_max = -7, 7
x2_min, x2_max = -7, 7
n_x1, n_x2 = 70, 70

p1_min, p1_max = -7, 7
p2_min, p2_max = -7, 7
n_p1, n_p2 = 70, 70

dx1 = (x1_max - x1_min) / (n_x1 - 1)
dx2 = (x2_max - x2_min) / (n_x2 - 1)
dp1 = (p1_max - p1_min) / (n_p1 - 1)
dp2 = (p2_max - p2_min) / (n_p2 - 1)


# Create coordinate grids
x1 = np.linspace(x1_min, x1_max, n_x1)
x2 = np.linspace(x2_min, x2_max, n_x2)
p1 = np.linspace(p1_min, p1_max, n_p1)
p2 = np.linspace(p2_min, p2_max, n_p2)


def Psi(x1,x2):
    Psi_0_0 = ((m * w / (np.pi * h_bar))**0.5)*(np.exp(-m * w*(x1**2+x2**2)/ (2 * h_bar)))
    Psi_0_1 = (np.sqrt(2/np.pi)*(m**3 * w**3 / (h_bar**3))**0.25)*x2*(np.exp(-m * w * (x1**2 + x2**2) / (2 * h_bar)))

    result = np.sqrt(3/5)*Psi_0_0 + np.sqrt(2/5)*Psi_0_1
    return result

Psi_star = lambda x1, x2: np.conj(Psi(x1, x2))


y_min, y_max = -7, 7
n_y = 100  # number of points for integration grid
y1 = np.linspace(y_min, y_max, n_y)
y2 = np.linspace(y_min, y_max, n_y)
Y1, Y2 = np.meshgrid(y1, y2, indexing='ij')


x1_val, x2_val, p1_val, p2_val = -1.52, -1.52, -1.52, -1.52


integrand_vals = np.real(
    Psi_star(x1_val + Y1, x2_val + Y2) *
    Psi(x1_val - Y1, x2_val - Y2) *
    np.exp(2j * (p1_val * Y1 + p2_val * Y2) / h_bar)
)

integral_y2 = simps(integrand_vals, y2, axis=1)
integral = simps(integral_y2, y1)
W_val = integral / ((np.pi * h_bar) ** 2)



print(f"\nWigner function at (x1,x2,p1,p2)=({x1_val},{x2_val},{p1_val},{p2_val}):")
print(W_val)



'''
# Create meshgrid for plotting
X, P = np.meshgrid(x, p, indexing='ij')
# Create 3D plot
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Surface plot
surf = ax.plot_surface(X, P, W, cmap='viridis', edgecolor='none')

# Labels
ax.set_xlabel('x')
ax.set_ylabel('p')
ax.set_zlabel('W(x, p)')
ax.set_title('Wigner Function at t = 0')

# Add color bar
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)

plt.tight_layout()
plt.show()
'''

