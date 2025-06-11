#import sys
#print(sys.executable)
import math
import numpy as np
import pandas as pd
from scipy.integrate import simpson as simps
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import time
np.set_printoptions(threshold=np.inf)
MAX_DIM = 4

m = 1
w = 1
h_bar = 1

x1_min, x1_max = -5, 5
x2_min, x2_max = -5, 5
n_x1, n_x2 = 70, 70

p1_min, p1_max = -5, 5
p2_min, p2_max = -5, 5
n_p1, n_p2 = 70, 70

dx1 = (x1_max - x1_min) / (n_x1 - 1)
dx2 = (x2_max - x2_min) / (n_x2 - 1)
dp1 = (p1_max - p1_min) / (n_p1 - 1)
dp2 = (p2_max - p2_min) / (n_p2 - 1)

N = 70 
spacing = np.linspace(-10, 10, N)
x,y,z,w = np.meshgrid(spacing, spacing, spacing, spacing)
dx_emd = spacing[1]-spacing[0] # spacing dx
tau = 3 # common safe default for 2D/ 3D problems
mu = 1./(16*tau*(N-1)**2) # ensures the operator norm of the gradient is controlled


t_f = 0.1  #final time
n_t = 70
dt = t_f/n_t
t = 0.1

x1 = np.linspace(x1_min, x1_max, n_x1)
x2 = np.linspace(x2_min, x2_max, n_x2)
p1 = np.linspace(p1_min, p1_max, n_p1)
p2 = np.linspace(p2_min, p2_max, n_p2)
X1,X2,P1,P2 = np.meshgrid(x1,x2,p1,p2,indexing='ij')
U = 0.5*m*w*w*(X1**2+X2**2)

def Psi_src(x1,x2):
    Psi_0_0 = ((m * w / (np.pi * h_bar))**0.5)*(np.exp(-m * w*(x1**2+x2**2)/ (2 * h_bar)))
    Psi_0_1 = (np.sqrt(2)*(m**3 * w**3 / (np.pi * h_bar**3))**0.25)*x2*(np.exp(-m * w * (x1**2 + x2**2) / (2 * h_bar)))
    result = np.sqrt(3/5)*Psi_0_0 + np.sqrt(2/5)*Psi_0_1
    return result
def Psi_dest(x1,x2):
    Psi_0_0 = ((m * w / (np.pi * h_bar))**0.5)*(np.exp(-m * w*(x1**2+x2**2)/ (2 * h_bar)))
    Psi_0_1 = (np.sqrt(2)*(m**3 * w**3 / (np.pi * h_bar**3))**0.25)*x2*(np.exp(-m * w * (x1**2 + x2**2) / (2 * h_bar)))
    result = np.sqrt(3.1/5)*Psi_0_0 + np.sqrt(1.9/5)*Psi_0_1
    return result
Psi_star_src = lambda x1, x2: np.conj(Psi_src(x1, x2))
Psi_star_dest = lambda x1, x2: np.conj(Psi_dest(x1, x2))

y_min, y_max = -5, 5
n_y = 70
y1_vals = np.linspace(y_min, y_max, n_y)
y2_vals = np.linspace(y_min, y_max, n_y)
Y1, Y2 = np.meshgrid(y1_vals, y2_vals, indexing='ij')

def compute_wigner_element(i, j, k, l,Psi_star,Psi):
    integrand_vals = np.real(
        Psi_star(x1[i] + Y1, x2[j] + Y2) *
        Psi(x1[i] - Y1, x2[j] - Y2) *
        np.exp(2j * (p1[k] * Y1 + p2[l] * Y2) / h_bar)
    )
    integral_y2 = simps(integrand_vals, y2_vals, axis=1)
    integral = simps(integral_y2, y1_vals)
    return i, j, k, l, integral / ((np.pi * h_bar) ** 2)

def Wigner(Psi,Psi_star):
    result = np.zeros((n_x1, n_x2, n_p1, n_p2))
    indices = [(i, j, k, l) for i in range(n_x1)
                            for j in range(n_x2)
                            for k in range(n_p1)
                            for l in range(n_p2)]
    results = Parallel(n_jobs=-1, verbose=0)(
        delayed(compute_wigner_element)(i, j, k, l,Psi_star,Psi) for (i, j, k, l) in indices
    )
    for i, j, k, l, val in results:
        result[i, j, k, l] = val

    return result
source = Wigner(Psi_src,Psi_star_src)
dest = Wigner(Psi_dest,Psi_star_dest)

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
        W[0:2, :, :, :] = W[-2:, :, :, :] = W[:, 0:2, :, :] = W[:, -2:, :, :] = W[:, :, 0:2, :] = W[:, :, -2:, :] = W[:, :, :, 0:2] = W[:, :, :, -2:] = 0
    return W

def l2_update(phi: np.ndarray, m: np.ndarray, m_temp: np.ndarray, rhodiff: np.ndarray, tau, mu, dx, dim):
    m_temp[:] = -m
    m[0, :-1] += mu * (phi[1:] - phi[:-1]) / dx
    if dim > 1:
        m[1, :, :-1] += mu * (phi[:, 1:] - phi[:, :-1]) / dx
    if dim > 2:
        m[2, :, :, :-1] += mu * (phi[:, :, 1:] - phi[:, :, :-1]) / dx
    if dim > 3:
        m[3, :, :, :, :-1] += mu * (phi[:, :, :, 1:] - phi[:, :, :, :-1]) / dx
    
    norm = np.sqrt(np.sum(m**2, axis=0))
    shrink_factor = 1 - mu / np.maximum(norm, mu)
    m *= shrink_factor[None, ...]
    m_temp += 2*m
    divergence = m_temp.sum(axis=0)
    divergence[1:] -= m_temp[0, :-1]
    if dim > 1: divergence[:, 1:] -= m_temp[1, :, :-1]
    if dim > 2: divergence[:, :, 1:] -= m_temp[2, :, :, :-1]
    if dim > 3: divergence[:, :, :, 1:] -= m_temp[3, :, :, :, :-1]
    
    phi += tau * (divergence/dx + rhodiff)

def l2_distance(source: np.ndarray, dest: np.ndarray, dx, maxiter=100000, tau=3, mu=3e-6):
    if len(source.shape) > MAX_DIM:
        raise ValueError(f"Dimensions of greater than {MAX_DIM} are not supported!")
    elif source.shape != dest.shape:
        raise ValueError(f"Dimension mismatch between source and destination! Source shape is '{source.shape}', dest shape is '{dest.shape}'.")
    rhodiff = np.array(dest-source)
    phi = np.zeros_like(rhodiff)
    m = np.zeros((len(phi.shape),) + phi.shape)
    m_temp = np.zeros_like(m)
    dim = len(phi.shape)
    for i in range(maxiter):
        l2_update(phi, m, m_temp, rhodiff, tau=tau, mu=mu, dx=dx, dim=dim)
        #if i %1000 == 0:
        #    print(f"Iteration: {i}, L2 distance", np.sum(np.sqrt(np.sum(m**2,axis=0))))
    return np.sum(np.sqrt(np.sum(m**2,axis=0)))

emd = []
start_time = time.time()
if __name__ == "__main__":

    source /= source.sum()
    dest /= dest.sum()

    computedDistance = l2_distance(source, dest, maxiter=40000, dx_emd=dx_emd, tau=tau, mu = mu)
    emd.append(computedDistance)
    print("Earth Mover's Distance at t = 0s:", computedDistance)

    for i in range (int(t/t_f)):

        source = timeEvol(source)
        dest = timeEvol(dest)

        source /= source.sum()
        dest /= dest.sum()

        computedDistance = l2_distance(source, dest, maxiter=40000, dx_emd=dx_emd, tau=tau, mu = mu)
        emd.append(computedDistance)
        print("Earth Mover's Distance at t = " + f"{(i + 1) * t_f:.1f}" + "s:", computedDistance)

    end_time = time.time()
    print(end_time-start_time)
    times = np.arange(0, t + t_f, t_f)
    plt.figure()
    plt.plot(times, emd, marker='o')
    plt.xlabel('Time (s)')
    plt.ylabel("Earth Mover's Distance")
    plt.title("EMD evolution over time")
    plt.grid(True)
    plt.show()