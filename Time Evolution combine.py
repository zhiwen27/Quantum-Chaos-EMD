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

t_f=0.1 #final time
n_t=1000     #Number of steps taken for time
dt=t_f/n_t

# Compute step sizes
dx = (x_max - x_min) / (n_x - 1)
dp = (p_max - p_min) / (n_p - 1)

# Create grid points
x = x_min + dx * np.arange(n_x)
p = p_min + dp * np.arange(n_p)
#print("\nArray of all x points:\n",x)

def Psi_src(x):
    Psi_0 = ((m * w / (np.pi * h_bar))**0.25)*(np.exp(-m * w * x**2 / (2 * h_bar)))
    Psi_1 = ((m * w / (np.pi * h_bar))**0.25)*(np.exp(-m * w * x**2 / (2 * h_bar))*(np.sqrt(2*m*w/h_bar))*x)
    result=np.sqrt(3/5)*Psi_0 + np.sqrt(2/5)*Psi_1
    return result
Psi_star_src=Psi_src

def Psi_dest(x):
    Psi_0 = ((m * w / (np.pi * h_bar))**0.25)*(np.exp(-m * w * x**2 / (2 * h_bar)))
    Psi_1 = ((m * w / (np.pi * h_bar))**0.25)*(np.exp(-m * w * x**2 / (2 * h_bar))*(np.sqrt(2*m*w/h_bar))*x)
    result=np.sqrt(3.1/5)*Psi_0 + np.sqrt(1.9/5)*Psi_1
    return result
Psi_star_dest=Psi_dest

# can do faster integration
def Wigner_src(Psi_func):
    result=np.zeros((n_x,n_p))
    for i in range(n_x):
        for j in range(n_p):
            integrand=lambda y: np.real(Psi_star_src(x[i]+y)*(Psi_src(x[i]-y))*np.exp(2j*p[j]*y/h_bar))
            integral_value, _ = quad(integrand,-np.inf,np.inf)
            result[i,j]=integral_value/(np.pi*h_bar)
    return result
def Wigner_dest(Psi_func):
    result=np.zeros((n_x,n_p))
    for i in range(n_x):
        for j in range(n_p):
            integrand=lambda y: np.real(Psi_star_dest(x[i]+y)*(Psi_dest(x[i]-y))*np.exp(2j*p[j]*y/h_bar))
            integral_value, _ = quad(integrand,-np.inf,np.inf)
            result[i,j]=integral_value/(np.pi*h_bar)
    return result
W_src=Wigner_src(Psi_src)
W_dest = Wigner_dest(Psi_dest)


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

def timeEvol(W):
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
    return W

np.savetxt("Wigner_init_src.txt", W_src, delimiter=" ")
np.savetxt("Wigner_init_dest.txt", W_dest, delimiter=" ")

W_src = timeEvol(W_src)
W_dest = timeEvol(W_dest)

np.savetxt("Wigner_final_src.txt", W_src, delimiter=" ")
np.savetxt("Wigner_final_dest.txt", W_dest, delimiter=" ")