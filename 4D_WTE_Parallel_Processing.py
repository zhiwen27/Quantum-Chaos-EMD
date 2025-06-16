import math
import numpy as np
import pandas as pd
from scipy.integrate import simpson as simps
from joblib import Parallel, delayed
import os
import multiprocessing
import time
start_time = time.time()


print("Here I start again!")

# ========== SYSTEM CHECK ==========
print("🧠 CPUs visible to Python:", multiprocessing.cpu_count())
print("🔧 CPUs allocated by SLURM:", os.environ.get("SLURM_CPUS_ON_NODE"))

m=1
w=1
h_bar=1

# Spatial and momentum grid parameters
x1_min, x1_max = -5,5
x2_min, x2_max = -5,5
n_x1, n_x2 = 50, 50

p1_min, p1_max = -5, 5
p2_min, p2_max = -5, 5
n_p1, n_p2 = 50, 50

dx1 = (x1_max - x1_min) / (n_x1 - 1)
dx2 = (x2_max - x2_min) / (n_x2 - 1)
dp1 = (p1_max - p1_min) / (n_p1 - 1)
dp2 = (p2_max - p2_min) / (n_p2 - 1)

t_f=0.1  #final time
n_t=100
dt=t_f/n_t

# Create coordinate grids
x1 = np.linspace(x1_min, x1_max, n_x1)
x2 = np.linspace(x2_min, x2_max, n_x2)
p1 = np.linspace(p1_min, p1_max, n_p1)
p2 = np.linspace(p2_min, p2_max, n_p2)


def Psi(x1,x2):
    Psi_0_0 = ((m * w / (np.pi * h_bar))**0.5)*(np.exp(-m * w*(x1**2+x2**2)/ (2 * h_bar)))
    Psi_0_1 = (np.sqrt(2/np.pi)* (m*w/(h_bar))**(3/4)) *x2 *(np.exp(-m * w * (x1**2 + x2**2) / (2 * h_bar)))
    result=np.sqrt(3/5)*Psi_0_0 + np.sqrt(2/5)*Psi_0_1
    return result
Psi_star = lambda x1, x2: np.conj(Psi(x1, x2))


y_min, y_max = -5, 5
n_y = 70  # number of points for integration grid
y1_vals = np.linspace(y_min, y_max, n_y)
y2_vals = np.linspace(y_min, y_max, n_y)


def compute_wigner_element(i, j, k, l):
    Y1, Y2 = np.meshgrid(y1_vals, y2_vals, indexing='ij')
    integrand_vals = np.real(
        Psi_star(x1[i] + Y1, x2[j] + Y2) *
        Psi(x1[i] - Y1, x2[j] - Y2) *
        np.exp(2j * (p1[k] * Y1 + p2[l] * Y2) / h_bar)
    )
    integral_y2 = simps(integrand_vals, y2_vals, axis=1)
    integral = simps(integral_y2, y1_vals)
    return i, j, k, l, integral / ((np.pi * h_bar) ** 2)

def Wigner(Psi_func):
    print("\nStarting parallel Wigner computation...")
    t0 = time.time()

    result = np.zeros((n_x1, n_x2, n_p1, n_p2))
    indices = [(i, j, k, l) for i in range(n_x1)
                            for j in range(n_x2)
                            for k in range(n_p1)
                            for l in range(n_p2)]

    results = Parallel(n_jobs=-1, verbose=0)(
        delayed(compute_wigner_element)(i, j, k, l) for (i, j, k, l) in indices
    )

    for i, j, k, l, val in results:
        result[i, j, k, l] = val

    t1 = time.time()
    wigner_time = t1 - t0
    print("✅ Parallel Wigner computation completed in {:.2f} seconds.\n".format(wigner_time))
    return result, wigner_time


W, wigner_time = Wigner(Psi)


X1,X2,P1,P2 = np.meshgrid(x1,x2,p1,p2,indexing='ij')  # full 2D grid versions of x and p  #Grid for X,P
U = 0.5*m*w*w*(X1**2+X2**2)


def central_diff_4th_order_parallel(W_array, axis, spacing, n_jobs=-1):
    result = np.zeros_like(W_array)
    shape = W_array.shape

    def compute_slice(i):
        slc1 = [slice(None)] * 4
        slc2 = [slice(None)] * 4
        slc3 = [slice(None)] * 4
        slc4 = [slice(None)] * 4
        center = [slice(None)] * 4

        slc1[axis] = i + 2
        slc2[axis] = i + 1
        slc3[axis] = i - 1
        slc4[axis] = i - 2
        center[axis] = i

        return (
            tuple(center),
            (-W_array[tuple(slc1)] + 8 * W_array[tuple(slc2)]
             - 8 * W_array[tuple(slc3)] + W_array[tuple(slc4)]) / (12 * spacing)
        )

    indices = range(2, shape[axis] - 2)

    results = Parallel(n_jobs=n_jobs)(
        delayed(compute_slice)(i) for i in indices
    )

    for center_slice, val in results:
        result[center_slice] = val

    return result

# Time derivative function for RK4
def f(W_array):
    dW_dx1 = central_diff_4th_order_parallel(W_array, axis=0, spacing=dx1)
    dW_dx2 = central_diff_4th_order_parallel(W_array, axis=1, spacing=dx2)
    dW_dp1 = central_diff_4th_order_parallel(W_array, axis=2, spacing=dp1)
    dW_dp2 = central_diff_4th_order_parallel(W_array, axis=3, spacing=dp2)

    return (
        (-1 / m) * P1 * dW_dx1 +
        (-1 / m) * P2 * dW_dx2 +
        m * w**2 * X1 * dW_dp1 +
        m * w**2 * X2 * dW_dp2
    )


# RK4 time evolution
print("🚀 Starting RK4 time evolution...")
rk4_start = time.time()
for step in range(n_t):

    k1=f(W)
    W1=W+k1*dt/2
    k2=f(W1)
    W2=W+k2*dt/2
    k3=f(W2)
    W3=W+k3*dt
    k4=f(W3)
    W=W+(dt/6)*(k1+2*k2+2*k3+k4)
    W[0:2, :, :, :] = W[-2:, :, :, :] = W[:, 0:2, :, :] = W[:, -2:, :, :] = W[:, :, 0:2, :] = W[:, :, -2:, :] = W[:, :, :, 0:2] = W[:, :, :, -2:] = 0
rk4_end = time.time()
rk4_time = rk4_end - rk4_start
print("✅ RK4 time evolution completed in {:.2f} seconds.\n".format(rk4_time))




print("n_boxes =", n_x1, "; n_y =", n_y, "; n_t =", n_t, "; Time,t =", t_f)

i1 = np.argmin(np.abs(x1 - (-1)))
j1 = np.argmin(np.abs(x2 - (-1)))
k1 = np.argmin(np.abs(p1 - (-1)))
l1 = np.argmin(np.abs(p2 - (-1)))
print("\nW_derived value at closest (x1=%.2f,x2=%.2f,p1=%.2f,p2=%.2f):" % (x1[i1],x2[j1],p1[k1],p2[l1]), W[i1,j1,k1,l1])

i2 = np.argmin(np.abs(x1 - (-0.7)))
j2 = np.argmin(np.abs(x2 - (-0.7)))
k2 = np.argmin(np.abs(p1 - (-0.7)))
l2 = np.argmin(np.abs(p2 - (-0.7)))
print("\nW_derived value at closest (x1=%.2f,x2=%.2f,p1=%.2f,p2=%.2f):" % (x1[i2],x2[j2],p1[k2],p2[l2]), W[i2,j2,k2,l2])

i3 = np.argmin(np.abs(x1 - (-0.5)))
j3 = np.argmin(np.abs(x2 - (-0.5)))
k3 = np.argmin(np.abs(p1 - (-0.5)))
l3 = np.argmin(np.abs(p2 - (-0.5)))
print("\nW_derived value at closest (x1=%.2f,x2=%.2f,p1=%.2f,p2=%.2f):" % (x1[i3],x2[j3],p1[k3],p2[l3]), W[i3,j3,k3,l3])

i4 = np.argmin(np.abs(x1 - (0)))
j4 = np.argmin(np.abs(x2 - (0)))
k4 = np.argmin(np.abs(p1 - (0)))
l4 = np.argmin(np.abs(p2 - (0)))
print("\nW_derived value at closest (x1=%.2f,x2=%.2f,p1=%.2f,p2=%.2f):" % (x1[i4],x2[j4],p1[k4],p2[l4]), W[i4,j4,k4,l4])

i5 = np.argmin(np.abs(x1 - (0.5)))
j5 = np.argmin(np.abs(x2 - (0.5)))
k5 = np.argmin(np.abs(p1 - (0.5)))
l5 = np.argmin(np.abs(p2 - (0.5)))
print("\nW_derived value at closest (x1=%.2f,x2=%.2f,p1=%.2f,p2=%.2f):" % (x1[i5],x2[j5],p1[k5],p2[l5]), W[i5,j5,k5,l5])

i6 = np.argmin(np.abs(x1 - (0.7)))
j6 = np.argmin(np.abs(x2 - (0.7)))
k6 = np.argmin(np.abs(p1 - (0.7)))
l6 = np.argmin(np.abs(p2 - (0.7)))
print("\nW_derived value at closest (x1=%.2f,x2=%.2f,p1=%.2f,p2=%.2f):" % (x1[i6],x2[j6],p1[k6],p2[l6]), W[i6,j6,k6,l6])

i7 = np.argmin(np.abs(x1 - (1)))
j7 = np.argmin(np.abs(x2 - (1)))
k7 = np.argmin(np.abs(p1 - (1)))
l7 = np.argmin(np.abs(p2 - (1)))
print("\nW_derived value at closest (x1=%.2f,x2=%.2f,p1=%.2f,p2=%.2f):" % (x1[i7],x2[j7],p1[k7],p2[l7]), W[i7,j7,k7,l7])



np.save("W_output.npy", W)


end_time = time.time()
total_time = end_time - start_time

print("\n⏱️ Summary:")
print(f" - Wigner conversion time: {wigner_time:.2f} seconds ({wigner_time / 60:.2f} minutes)")
print(f" - RK4 time evolution time: {rk4_time:.2f} seconds ({rk4_time / 60:.2f} minutes)")
print(f" - Total runtime: {total_time:.2f} seconds ({total_time / 60:.2f} minutes)")
