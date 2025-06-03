import math
import numpy as np
#import pandas as pd
from scipy.integrate import quad
#import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D  # for 3D plots


m=1
w=1
h_bar=1

# Define the domain boundaries and number of boxes
x_min = -6
x_max = 6
n_x = 101   # Number of boxes (points = boxes)
p_min = -6
p_max = 6
n_p = 101

t_f=0.1  #final time
n_t=100 #Number of steps taken for time
dt=t_f/n_t

# Compute step sizes
dx = (x_max - x_min) / (n_x - 1)
dp = (p_max - p_min) / (n_p - 1)

# Create grid points
x = x_min + dx * np.arange(n_x)
p = p_min + dp * np.arange(n_p)
#print("\nArray of all x points:\n",x)

def Psi(x):
    Psi_0 = ((m * w / (np.pi * h_bar))**0.25)*(np.exp(-m * w * x**2 / (2 * h_bar)))
    Psi_1 = ((m * w / (np.pi * h_bar))**0.25)*(np.exp(-m * w * x**2 / (2 * h_bar))*(np.sqrt(2*m*w/h_bar))*x)
    result=np.sqrt(3/5)*Psi_0 + np.sqrt(2/5)*Psi_1
    return result
Psi_star=Psi


def Wigner(Psi_func):
    result=np.zeros((n_x,n_p))
    for i in range(n_x):
        for j in range(n_p):
            integrand=lambda y: np.real(Psi_star(x[i]+y)*(Psi(x[i]-y))*np.exp(2j*p[j]*y/h_bar))
            integral_value, _ = quad(integrand,-np.inf,np.inf)
            result[i,j]=integral_value/(np.pi*h_bar)
    return result
W=Wigner(Psi)


U = 0.5*m*w*w*x**2
X, P = np.meshgrid(x, p,indexing='ij')  # full 2D grid versions of x and p  #Grid for X,P


def f(W_array):
    result=np.zeros((n_x,n_p))

    def xW_dev(W_array):
        result=np.zeros((n_x,n_p))
        for i in range(2,n_x-2):
            for j in range(2,n_p-2):
                # Using 4th-order central differences
                result[i, j] = (-W_array[i + 2, j] + 8 * W_array[i + 1, j]- 8 * W_array[i - 1, j] + W_array[i - 2, j]) / (12 * dx)
        return result
    xW_p=xW_dev(W_array)

    def pW_dev(W_array):
        result=np.zeros((n_x,n_p))
        for i in range(2,n_x-2):
            for j in range(2,n_p-2):
                # Using 4th-order central differences
                result[i, j] = (-W_array[i, j + 2] + 8 * W_array[i, j + 1]- 8 * W_array[i, j - 1] + W_array[i, j - 2]) / (12 * dp)
        return result
    pW_p=pW_dev(W_array)

    result[2:-2,2:-2] = ((-1/m)*(P[2:-2,2:-2])*(xW_p[2:-2,2:-2])) + (m*(w**2)*(X[2:-2,2:-2])*(pW_p[2:-2,2:-2]))
    return result

#Grid for W_dot
for step in range(n_t):

    k1=f(W)
    W1=W+k1*dt/2
    k2=f(W1)
    W2=W+k2*dt/2
    k3=f(W2)
    W3=W+k3*dt
    k4=f(W3)
    W=W+(dt/6)*(k1+2*k2+2*k3+k4)
    W[0:2, :] = W[-2:, :] = W[:, 0:2] = W[:, -2:] = 0  # Zero the boundary


i1 = np.argmin(np.abs(x - (-2)))
j1 = np.argmin(np.abs(p - (-2)))
print("\nW_derived value at closest (x=%.2f,p=%.2f):" % (x[i1], p[j1]), W[i1, j1])

i2 = np.argmin(np.abs(x - (-1)))
j2 = np.argmin(np.abs(p - (-1)))
print("\nW_derived value at closest (x=%.2f,p=%.2f):" % (x[i2], p[j2]), W[i2, j2])

i3 = np.argmin(np.abs(x - (-0.5)))
j3 = np.argmin(np.abs(p - (-0.5)))
print("\nW_derived value at closest (x=%.2f,p=%.2f):" % (x[i3], p[j3]), W[i3, j3])

i4 = np.argmin(np.abs(x - (0.5)))
j4 = np.argmin(np.abs(p - (0.5)))
print("\nW_derived value at closest (x=%.2f,p=%.2f):" % (x[i4], p[j4]), W[i4, j4])

i5 = np.argmin(np.abs(x - (1)))
j5 = np.argmin(np.abs(p - (1)))
print("\nW_derived value at closest (x=%.2f,p=%.2f):" % (x[i5], p[j5]), W[i5, j5])

i6 = np.argmin(np.abs(x - (2)))
j6 = np.argmin(np.abs(p - (2)))
print("\nW_derived value at closest (x=%.2f,p=%.2f):" % (x[i6], p[j6]), W[i6, j6])

i7 = np.argmin(np.abs(x - (-0.58)))
j7 = np.argmin(np.abs(p - (-0.58)))
print("\nW_derived value at closest (x=%.2f,p=%.2f):" % (x[i7], p[j7]), W[i7, j7])





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
ax.set_title('Wigner Function at t = 0.1')

# Add color bar
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)

plt.tight_layout()
plt.show()


print("\nGrid of x values:\n",X)
print("\nGrid of p values:\n",P)

DW=pd.DataFrame(W,index=x,columns=p)
print("\nGrid of W at t=0:\n",DW)

plt.contourf(x, p, W.T, levels=50, cmap='RdBu')
plt.xlabel('x')
plt.ylabel('p')
plt.title('Wigner Function at final time')
plt.colorbar()
plt.show()


DW_dot = pd.DataFrame(W_dot, index=x, columns=p)
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", None)
#print("\nGrid of W_dot at t=0:\n",DW_dot)
#DxW_p = pd.DataFrame(xW_p, index=x, columns=p)
#Dypsi_p = pd.DataFrame(pW_p, index=x, columns=p)
#Dpsi_dot = pd.DataFrame(W_dot, index=x, columns=p)
#print("\nFinal W at t=0.1\n:", DW)
#print("\nFinal xW_p at t=0.1\n:", DxW_p)
#print("\nFinal pW_p at t=0.1\n:",DpW_p)
#print("\nFinal W_dot at t=0.1\n:",DW_dot)
'''
