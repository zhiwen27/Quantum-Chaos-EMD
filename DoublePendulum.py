import math
import numpy as np
import pandas as pd
from scipy.integrate import simpson as simps
from joblib import Parallel, delayed
import os
import multiprocessing
import time
from scipy.interpolate import RectBivariateSpline
total_start = time.time()

print("Here I start again!")

# ========== SYSTEM CHECK ==========
print("🧠 CPUs visible to Python:", multiprocessing.cpu_count())
print("🔧 CPUs allocated by SLURM:", os.environ.get("SLURM_CPUS_ON_NODE"))

m1=1000
m2=1
g=10
l1=1
l2=10
I1=m1*l1**2
I2=m2*l2**2
w1=np.sqrt(g/l1)
w2=np.sqrt(g/l2)
h_bar=1
u1=0
u2=0
f1=0.0001
f2=0.1


a = (m1+m2)*l1 + m2*l2**2
b = m2*l1*l2
c = m2*l2

# Spatial and momentum grid parameters
o1_min, o1_max = -0.08,0.08
o2_min, o2_max = -0.8,0.8
n_o1, n_o2 = 161, 161

n1 = np.arange(-60, 61)
n2 = np.arange(-6, 7)

n_y = 200  # number of points in y grid for integration
y1_vals = np.linspace(-np.pi/2, np.pi/2, n_y)
y2_vals = np.linspace(-np.pi/2, np.pi/2, n_y)

dO1 = (o1_max - o1_min) / (n_o1 - 1)
dO2 = (o2_max - o2_min) / (n_o2 - 1)


t_f=0  #final time
dt=0.0001
n_t=math.ceil(t_f/dt) #Round Up

# Create coordinate grids
o1 = np.linspace(o1_min, o1_max, n_o1)
o2 = np.linspace(o2_min, o2_max, n_o2)

D = (-h_bar**2/ 2)*(1/((a*c)-(c**2)-(b**2)*(np.cos(o2)**2)))
A = c*D
B = a*D + 2*b*D*np.cos(o2)
C = 2*c*D + 2*b*D*np.cos(o2)


def Psi(O1,O2,f1,f2,u1,u2):
    G = (1/(np.sqrt(2*np.pi))*f1*f2) * np.exp(-((O1-u1)**2/2*f1**2)-((O2-u2)**2/2*f2**2))
    result= G
    return result


O1, O2 = np.meshgrid(o1, o2, indexing='ij')  # full 2D grid versions of x and p  #Grid for X,P
P = Psi(O1,O2,f1,f2,u1,u2).astype(complex)
U = -m1*g*l1*np.cos(O1) - m2*g*(l1*np.cos(O1) + l2*np.cos(O2))

def second_derivative_4th_order_parallel(P_array, axis, spacing, n_jobs=-1):
    result = np.zeros_like(P_array, dtype=complex)
    shape = P_array.shape

    def compute_slice(i):
        slc1 = [slice(None)] * 2
        slc2 = [slice(None)] * 2
        slc3 = [slice(None)] * 2
        slc4 = [slice(None)] * 2
        center = [slice(None)] * 2

        slc1[axis] = i - 2
        slc2[axis] = i - 1
        slc3[axis] = i + 1
        slc4[axis] = i + 2
        center[axis] = i

        val = (-P_array[tuple(slc4)] + 16*P_array[tuple(slc3)] - 30*P_array[tuple(center)] +
               16*P_array[tuple(slc2)] - P_array[tuple(slc1)]) / (12 * spacing**2)
        return (tuple(center), val)

    indices = range(2, shape[axis] - 2)
    results = Parallel(n_jobs=n_jobs)(delayed(compute_slice)(i) for i in indices)

    for center_slice, val in results:
        result[center_slice] = val

    return result


def mixed_second_derivative_4th_order_parallel(P_array, dx1, dx2, n_jobs=-1):
    result = np.zeros_like(P_array, dtype=complex)
    shape = P_array.shape

    def compute_point(i, j):
        val = (
            +   P_array[i-2,j-2] - 8*P_array[i-2,j-1] + 8*P_array[i-2,j+1] -   P_array[i-2,j+2]
            - 8*P_array[i-1,j-2] +64*P_array[i-1,j-1] -64*P_array[i-1,j+1] + 8*P_array[i-1,j+2]
            + 8*P_array[i+1,j-2] -64*P_array[i+1,j-1] +64*P_array[i+1,j+1] - 8*P_array[i+1,j+2]
            -   P_array[i+2,j-2] + 8*P_array[i+2,j-1] - 8*P_array[i+2,j+1] +   P_array[i+2,j+2]
        )
        return (i, j, val / (144 * dx1 * dx2))

    # Only compute for valid interior points
    indices = [(i, j) for i in range(2, shape[0] - 2) for j in range(2, shape[1] - 2)]

    results = Parallel(n_jobs=n_jobs)(delayed(compute_point)(i, j) for i, j in indices)

    for i, j, val in results:
        result[i, j] = val

    return result


# Time derivative function for RK4
def f(P_array):
    d2P_dO1 = second_derivative_4th_order_parallel(P_array, axis=0, spacing=dO1)
    d2P_dO2 = second_derivative_4th_order_parallel(P_array, axis=1, spacing=dO2)
    d2P_dO1_dO2 = mixed_second_derivative_4th_order_parallel(P_array, dO1, dO2)

    H = A*d2P_dO1 + B*d2P_dO2 - C*d2P_dO1_dO2 + U
    return (H*P_array/(1j*h_bar))



# RK4 time evolution
print("🚀 Starting RK4 time evolution...")
rk4_start = time.time()
for step in range(n_t):

    k1=f(P)
    P1=P+k1*dt/2
    k2=f(P1)
    P2=P+k2*dt/2
    k3=f(P2)
    P3=P+k3*dt
    k4=f(P3)
    P=P+(dt/6)*(k1+2*k2+2*k3+k4)
    P[:2, :] = P[-2:, :] = P[:, :2] = P[:, -2:] = 0
rk4_end = time.time()
rk4_time = rk4_end - rk4_start


#Wigner Conversion
def Wigner(P, o1, o2, n1, n2, y1_vals, y2_vals, h_bar=1):
    t0 = time.time()
    result = np.zeros((len(o1), len(o2), len(n1), len(n2)), dtype=complex)

    # Create interpolator for Psi (works for complex functions)
    interp = RectBivariateSpline(o1, o2, P)

    def compute_element(i, j, k, l):
        o1_val = o1[i]
        o2_val = o2[j]
        n1_val = n1[k]
        n2_val = n2[l]

        Y1, Y2 = np.meshgrid(y1_vals, y2_vals, indexing='ij')

        # Calculate ψ*(a1+y1, a2+y2) * ψ(a1-y1, a2-y2) * exp(2i(n1y1 + n2y2))
        psi_plus = interp(o1_val + Y1, o2_val + Y2, grid=False)
        psi_minus = interp(o1_val - Y1, o2_val - Y2, grid=False)
        integrand = np.conj(psi_plus) * psi_minus * np.exp(2j*(n1_val*Y1 + n2_val*Y2))

        # Double integration
        integral_y2 = simps(integrand, y2_vals, axis=1)
        integral = simps(integral_y2, y1_vals)

        norm = 1 / (np.pi * h_bar)**2
        return (i, j, k, l, norm * integral)

    # Parallel computation
    indices = [(i,j,k,l) for i in range(len(o1))
                        for j in range(len(o2))
                        for k in range(len(n1))
                        for l in range(len(n2))]

    results = Parallel(n_jobs=-1, verbose=0)(
        delayed(compute_element)(*idx) for idx in indices
    )

    for i, j, k, l, val in results:
        result[i, j, k, l] = val

    print(f"\n✅ Wigner computed in {time.time()-t0:.2f}s")
    return result.real, time.time()-t0  # Return real part as Wigner function should be real
W, wigner_time = Wigner(P, o1, o2, n1, n2, y1_vals, y2_vals, h_bar=1)


print("n_boxes =", n_o1, "; n_y =", n_y, "; n_t =", n_t, "; Time,t =", t_f)

total_end = time.time()
total_time = total_end - total_start
print("\n⏱️ Summary:")
print(f" - RK4 time evolution time: {rk4_time:.2f} seconds ({rk4_time / 60:.2f} minutes)")
#print(f" - Wigner conversion time: {wigner_time:.2f} seconds ({wigner_time / 60:.2f} minutes)")
#print(f" - Total runtime: {total_time:.2f} seconds ({total_time / 60:.2f} minutes)")
