import math
import numpy as np
import pandas as pd
from scipy.integrate import simpson as simps
from joblib import Parallel, delayed
import os
import multiprocessing
import time
total_start = time.time()

print("Here I start again!")

# ========== SYSTEM CHECK ==========
print("🧠 CPUs visible to Python:", multiprocessing.cpu_count())
print("🔧 CPUs allocated by SLURM:", os.environ.get("SLURM_CPUS_ON_NODE"))

m1=10000
m2=10
g=10
l1=0.01
l2=1
w=np.sqrt(g/l2)
u1=0
u2=0.01
h_bar=1
f1=0.0001
f2=0.1


a = (m1+m2)*l1 + m2*l2**2
b = m2*l1*l2
c = m2*l2

# Spatial and momentum grid parameters
o1_min, o1_max = -3,3
o2_min, o2_max = -3,3
n_o1, n_o2 = 50, 50

p1_min, p1_max = -3,3
p2_min, p2_max = -3,3
n_p1, n_p2 = 50, 50

y_min, y_max = -5, 5
n_y = 100  # number of points for integration grid
y1_vals = np.linspace(y_min, y_max, n_y)
y2_vals = np.linspace(y_min, y_max, n_y)

dO1 = (o1_max - o1_min) / (n_o1 - 1)
dO2 = (o2_max - o2_min) / (n_o2 - 1)
dp1 = (p1_max - p1_min) / (n_p1 - 1)
dp2 = (p2_max - p2_min) / (n_p2 - 1)

t_f=0  #final time
dt=0.0001
n_t=math.ceil(t_f/dt) #Round Up

# Create coordinate grids
o1 = np.linspace(o1_min, o1_max, n_o1)
o2 = np.linspace(o2_min, o2_max, n_o2)
p1 = np.linspace(p1_min, p1_max, n_p1)
p2 = np.linspace(p2_min, p2_max, n_p2)

D = (-h_bar**2/ 2)*(1/((a*c)-(c**2)-(b**2)*(np.cos(o2)**2)))
A = c*D
B = a*D + 2*b*D*np.cos(o2)
C = 2*c*D + 2*b*D*np.cos(o2)

U = -m1*g*l1*np.cos(o1) - m2*g*(l1*np.cos(o1) + l2*np.cos(o2))

def Psi(O1,O2,f1,f2,u1,u2):
    G = (1/(np.sqrt(2*np.pi))*f1*f2) * np.exp(-((O1-u1)**2/2*f1**2)-((O2-u2)**2/2*f2**2))
    result= G
    return result
#Psi_star = lambda x1, x2: np.conj(Psi(x1, x2))


O1, O2 = np.meshgrid(o1, o2, indexing='ij')  # full 2D grid versions of x and p  #Grid for X,P
P = Psi(O1,O2,f1,f2,u1,u2).astype(complex)


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


'''
#Wigner Conversion
def Wigner(P_grid, x1, x2, p1, p2, y1_vals, y2_vals, h_bar=1):
    print("\n🎬 Starting parallel Wigner computation...")
    t0 = time.time()

    n_x1, n_x2 = len(x1), len(x2)
    n_p1, n_p2 = len(p1), len(p2)

    result = np.zeros((n_x1, n_x2, n_p1, n_p2))

    def compute_element(i, j, k, l):
        Y1, Y2 = np.meshgrid(y1_vals, y2_vals, indexing='ij')

        # Shifted indices (simplified assumption: Psi constant around grid point)
        psi_plus  = P_grid[i, j]  # Ideally: interpolated at (x1[i] + Y1, x2[j] + Y2)
        psi_minus = P_grid[i, j]  # Here we avoid full interpolation for speed

        integrand = np.real(
            np.conj(psi_plus) * psi_minus *
            np.exp(2j * (p1[k] * Y1 + p2[l] * Y2) / h_bar)
        )

        integral_y2 = simps(integrand, y2_vals, axis=1)
        integral = simps(integral_y2, y1_vals)
        return (i, j, k, l, integral / ((np.pi * h_bar) ** 2))

    indices = [(i, j, k, l) for i in range(n_x1)
                                for j in range(n_x2)
                                for k in range(n_p1)
                                for l in range(n_p2)]

    results = Parallel(n_jobs=-1, verbose=0)(
        delayed(compute_element)(i, j, k, l) for (i, j, k, l) in indices
    )

    for i, j, k, l, val in results:
        result[i, j, k, l] = val

    t1 = time.time()
    wigner_time = t1 - t0
    print("✅ Parallel Wigner computation completed in {:.2f} seconds.\n".format(wigner_time))
    return result, wigner_time

W, wigner_time = Wigner(P, o1, o2, p1, p2, y1_vals, y2_vals, h_bar=1)
'''


print("n_boxes =", n_o1, "; n_y =", n_y, "; n_t =", n_t, "; Time,t =", t_f)

#np.save("W_output.npy", W)

i1 = np.argmin(np.abs(o1 - 0))
j1 = np.argmin(np.abs(o2 - 0.2))
print("\nP_derived value at closest (x1=%.2f, x2=%.2f):" % (o1[i1], o2[j1]), P[i1, j1])

i1 = np.argmin(np.abs(o1 - 0))
j1 = np.argmin(np.abs(o2 - 1))
print("\nP_derived value at closest (x1=%.2f, x2=%.2f):" % (o1[i1], o2[j1]), P[i1, j1])

i1 = np.argmin(np.abs(o1 - (0)))
j1 = np.argmin(np.abs(o2 - (-0.2)))
print("\nP_derived value at closest (x1=%.2f, x2=%.2f):" % (o1[i1], o2[j1]), P[i1, j1])

i1 = np.argmin(np.abs(o1 - (0)))
j1 = np.argmin(np.abs(o2 - (-1)))
print("\nP_derived value at closest (x1=%.2f, x2=%.2f):" % (o1[i1], o2[j1]), P[i1, j1])

'''
i1 = np.argmin(np.abs(o1 - (-0.4)))
j1 = np.argmin(np.abs(o2 - (-0.4)))
k1 = np.argmin(np.abs(p1 - (0)))
l1 = np.argmin(np.abs(p2 - (0)))
print("\nW_derived value at closest (x1=%.2f,x2=%.2f,p1=%.2f,p2=%.2f):" % (o1[i1],o2[j1],p1[k1],p2[l1]), W[i1,j1,k1,l1])

i2 = np.argmin(np.abs(o1 - (0.4)))
j2 = np.argmin(np.abs(o2 - (0.4)))
k2 = np.argmin(np.abs(p1 - (0)))
l2 = np.argmin(np.abs(p2 - (0)))
print("\nW_derived value at closest (x1=%.2f,x2=%.2f,p1=%.2f,p2=%.2f):" % (o1[i2],o2[j2],p1[k2],p2[l2]), W[i2,j2,k2,l2])

i3 = np.argmin(np.abs(o1 - (-0.2)))
j3 = np.argmin(np.abs(o2 - (-0.2)))
k3 = np.argmin(np.abs(p1 - (0)))
l3 = np.argmin(np.abs(p2 - (0)))
print("\nW_derived value at closest (x1=%.2f,x2=%.2f,p1=%.2f,p2=%.2f):" % (o1[i3],o2[j3],p1[k3],p2[l3]), W[i3,j3,k3,l3])

i4 = np.argmin(np.abs(o1 - (0.1)))
j4 = np.argmin(np.abs(o2 - (0.1)))
k4 = np.argmin(np.abs(p1 - (0)))
l4 = np.argmin(np.abs(p2 - (0)))
print("\nW_derived value at closest (x1=%.2f,x2=%.2f,p1=%.2f,p2=%.2f):" % (o1[i4],o2[j4],p1[k4],p2[l4]), W[i4,j4,k4,l4])

i5 = np.argmin(np.abs(o1 - (-0.1)))
j5 = np.argmin(np.abs(o2 - (-0.1)))
k5 = np.argmin(np.abs(p1 - (0)))
l5 = np.argmin(np.abs(p2 - (0)))
print("\nW_derived value at closest (x1=%.2f,x2=%.2f,p1=%.2f,p2=%.2f):" % (o1[i5],o2[j5],p1[k5],p2[l5]), W[i5,j5,k5,l5])

i6 = np.argmin(np.abs(o1 - (-1)))
j6 = np.argmin(np.abs(o2 - (0)))
k6 = np.argmin(np.abs(p1 - (-0.5)))
l6 = np.argmin(np.abs(p2 - (0)))
print("\nW_derived value at closest (x1=%.2f,x2=%.2f,p1=%.2f,p2=%.2f):" % (o1[i6],o2[j6],p1[k6],p2[l6]), W[i6,j6,k6,l6])

i7 = np.argmin(np.abs(o1 - (-0.8)))
j7 = np.argmin(np.abs(o2 - (0)))
k7 = np.argmin(np.abs(p1 - (-0.5)))
l7 = np.argmin(np.abs(p2 - (0)))
print("\nW_derived value at closest (x1=%.2f,x2=%.2f,p1=%.2f,p2=%.2f):" % (o1[i7],o2[j7],p1[k7],p2[l7]), W[i7,j7,k7,l7])

i8 = np.argmin(np.abs(o1 - (-0.5)))
j8 = np.argmin(np.abs(o2 - (0)))
k8 = np.argmin(np.abs(p1 - (-0.5)))
l8 = np.argmin(np.abs(p2 - (0)))
print("\nW_derived value at closest (x1=%.2f,x2=%.2f,p1=%.2f,p2=%.2f):" % (o1[i8],o2[j8],p1[k8],p2[l8]), W[i8,j8,k8,l8])

i9 = np.argmin(np.abs(o1 - (-0.1)))
j9 = np.argmin(np.abs(o2 - (0)))
k9 = np.argmin(np.abs(p1 - (-0.1)))
l9 = np.argmin(np.abs(p2 - (0)))
print("\nW_derived value at closest (x1=%.2f,x2=%.2f,p1=%.2f,p2=%.2f):" % (o1[i9],o2[j9],p1[k9],p2[l9]), W[i9,j9,k9,l9])

i10 = np.argmin(np.abs(o1 - (0.1)))
j10 = np.argmin(np.abs(o2 - (0)))
k10 = np.argmin(np.abs(p1 - (0.1)))
l10 = np.argmin(np.abs(p2 - (0)))
print("\nW_derived value at closest (x1=%.2f,x2=%.2f,p1=%.2f,p2=%.2f):" % (o1[i10],o2[j10],p1[k10],p2[l10]), W[i10,j10,k10,l10])

i11 = np.argmin(np.abs(o1 - (0.3)))
j11 = np.argmin(np.abs(o2 - (0)))
k11 = np.argmin(np.abs(p1 - (0.3)))
l11 = np.argmin(np.abs(p2 - (0)))
print("\nW_derived value at closest (x1=%.2f,x2=%.2f,p1=%.2f,p2=%.2f):" % (o1[i11],o2[j11],p1[k11],p2[l11]), W[i11,j11,k11,l11])

i12 = np.argmin(np.abs(o1 - (0.5)))
j12 = np.argmin(np.abs(o2 - (0)))
k12 = np.argmin(np.abs(p1 - (0.5)))
l12 = np.argmin(np.abs(p2 - (0)))
print("\nW_derived value at closest (x1=%.2f,x2=%.2f,p1=%.2f,p2=%.2f):" % (o1[i12],o2[j12],p1[k12],p2[l12]), W[i12,j12,k12,l12])

i7 = np.argmin(np.abs(o1 - (0)))
j7 = np.argmin(np.abs(o2 - (1)))
k7 = np.argmin(np.abs(p1 - (0)))
l7 = np.argmin(np.abs(p2 - (0.5)))
print("\nW_derived value at closest (x1=%.2f,x2=%.2f,p1=%.2f,p2=%.2f):" % (o1[i7],o2[j7],p1[k7],p2[l7]), W[i7,j7,k7,l7])

i7 = np.argmin(np.abs(o1 - (0)))
j7 = np.argmin(np.abs(o2 - (-1)))
k7 = np.argmin(np.abs(p1 - (0)))
l7 = np.argmin(np.abs(p2 - (-1)))
print("\nW_derived value at closest (x1=%.2f,x2=%.2f,p1=%.2f,p2=%.2f):" % (o1[i7],o2[j7],p1[k7],p2[l7]), W[i7,j7,k7,l7])

i7 = np.argmin(np.abs(o1 - (0)))
j7 = np.argmin(np.abs(o2 - (1)))
k7 = np.argmin(np.abs(p1 - (0)))
l7 = np.argmin(np.abs(p2 - (0.5)))
print("\nW_derived value at closest (x1=%.2f,x2=%.2f,p1=%.2f,p2=%.2f):" % (o1[i7],o2[j7],p1[k7],p2[l7]), W[i7,j7,k7,l7])

i7 = np.argmin(np.abs(o1 - (0)))
j7 = np.argmin(np.abs(o2 - (-1)))
k7 = np.argmin(np.abs(p1 - (0)))
l7 = np.argmin(np.abs(p2 - (-1)))
print("\nW_derived value at closest (x1=%.2f,x2=%.2f,p1=%.2f,p2=%.2f):" % (o1[i7],o2[j7],p1[k7],p2[l7]), W[i7,j7,k7,l7])

i7 = np.argmin(np.abs(o1 - (0)))
j7 = np.argmin(np.abs(o2 - (-0.5)))
k7 = np.argmin(np.abs(p1 - (0)))
l7 = np.argmin(np.abs(p2 - (-0.5)))
print("\nW_derived value at closest (x1=%.2f,x2=%.2f,p1=%.2f,p2=%.2f):" % (o1[i7],o2[j7],p1[k7],p2[l7]), W[i7,j7,k7,l7])

i7 = np.argmin(np.abs(o1 - (0)))
j7 = np.argmin(np.abs(o2 - (0.5)))
k7 = np.argmin(np.abs(p1 - (0)))
l7 = np.argmin(np.abs(p2 - (0.5)))
print("\nW_derived value at closest (x1=%.2f,x2=%.2f,p1=%.2f,p2=%.2f):" % (o1[i7],o2[j7],p1[k7],p2[l7]), W[i7,j7,k7,l7])

i7 = np.argmin(np.abs(o1 - (0)))
j7 = np.argmin(np.abs(o2 - (0.3)))
k7 = np.argmin(np.abs(p1 - (0)))
l7 = np.argmin(np.abs(p2 - (0.3)))
print("\nW_derived value at closest (x1=%.2f,x2=%.2f,p1=%.2f,p2=%.2f):" % (o1[i7],o2[j7],p1[k7],p2[l7]), W[i7,j7,k7,l7])

i7 = np.argmin(np.abs(o1 - (0)))
j7 = np.argmin(np.abs(o2 - (0.1)))
k7 = np.argmin(np.abs(p1 - (0)))
l7 = np.argmin(np.abs(p2 - (-0.1)))
print("\nW_derived value at closest (x1=%.2f,x2=%.2f,p1=%.2f,p2=%.2f):" % (o1[i7],o2[j7],p1[k7],p2[l7]), W[i7,j7,k7,l7])
'''

total_end = time.time()
total_time = total_end - total_start
print("\n⏱️ Summary:")
print(f" - RK4 time evolution time: {rk4_time:.2f} seconds ({rk4_time / 60:.2f} minutes)")
#print(f" - Wigner conversion time: {wigner_time:.2f} seconds ({wigner_time / 60:.2f} minutes)")
#print(f" - Total runtime: {total_time:.2f} seconds ({total_time / 60:.2f} minutes)")
