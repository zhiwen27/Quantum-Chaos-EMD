import math
import numpy as np
import pandas as pd
from scipy.integrate import simpson as simps
#import sys
#print(sys.executable)
from joblib import Parallel, delayed
import time

np.set_printoptions(threshold=np.inf)
MAX_DIM = 4

start_time = time.time()
m=1
w=1
h_bar=1

# Spatial and momentum grid parameters
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

t_f=0.1  #final time
n_t=70
dt=t_f/n_t

# Create coordinate grids
x1 = np.linspace(x1_min, x1_max, n_x1)
x2 = np.linspace(x2_min, x2_max, n_x2)
p1 = np.linspace(p1_min, p1_max, n_p1)
p2 = np.linspace(p2_min, p2_max, n_p2)


def Psi(x1,x2):
    Psi_0_0 = ((m * w / (np.pi * h_bar))**0.5)*(np.exp(-m * w*(x1**2+x2**2)/ (2 * h_bar)))
    Psi_0_1 = (np.sqrt(2)*(m**3 * w**3 / (np.pi * h_bar**3))**0.25)*x2*(np.exp(-m * w * (x1**2 + x2**2) / (2 * h_bar)))
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
    print("Parallel Wigner computation completed in {:.2f} seconds.\n".format(t1 - t0))
    return result


W=Wigner(Psi)


X1,X2,P1,P2 = np.meshgrid(x1,x2,p1,p2,indexing='ij')  # full 2D grid versions of x and p  #Grid for X,P
U = 0.5*m*w*w*(X1**2+X2**2)

# 4th-order central derivative function
# Parallel central difference
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


def l2_update(phi: np.ndarray, m: np.ndarray, m_temp: np.ndarray, rhodiff: np.ndarray, tau, mu, dx, dim):
    """Do an L2 update."""

    m_temp[:] = -m # store previous state of m for divergence calculation (copy old m into m_temp)
    m[0, :-1] += mu * (phi[1:] - phi[:-1]) / dx # update in the x-direction; take all rows except the last (0 ~ N-2)
                                                # rows 1 ~ N-1 - rows 0 ~ N-2 (ϕ(x+Δx)−ϕ(x))/ dx
                                                # grid size uniform, only define dx
                                                # m: transport field, updated using the gradient of the dual potential ϕ
                                                # P2, eq 3
    if dim > 1:
        m[1, :, :-1] += mu * (phi[:, 1:] - phi[:, :-1]) / dx # update in the y-direction; for dim = 2: m = m[2 (dim) : 256 : 256]
    if dim > 2:
        m[2, :, :, :-1] += mu * (phi[:, :, 1:] - phi[:, :, :-1]) / dx # update in the z-direction
    if dim > 3:
        m[3, :, :, :, :-1] += mu * (phi[:, :, :, 1:] - phi[:, :, :, :-1]) / dx # update in the 4th-direction
    
    norm = np.sqrt(np.sum(m**2, axis=0)) # sum over all the m vectors; 2D array
    shrink_factor = 1 - mu / np.maximum(norm, mu) # from the solution to P8, eq 9; 2D array
    #shrink_factor[norm < mu] = 0
    m *= shrink_factor[None, ...] # None: add a new dim; multiply m by the shrink factor
    m_temp += 2*m # store the after state of m for divergence calculation
    divergence = m_temp.sum(axis=0) # summing up all m_temp vectors
    divergence[1:] -= m_temp[0, :-1] # computes m_temp[i] − m_temp[i−1] (x)
    if dim > 1: divergence[:, 1:] -= m_temp[1, :, :-1] # computes m_temp[i] − m_temp[i−1] (y)
    if dim > 2: divergence[:, :, 1:] -= m_temp[2, :, :, :-1] # computes m_temp[i] − m_temp[i−1] (z)
    if dim > 3: divergence[:, :, :, 1:] -= m_temp[3, :, :, :, :-1] # computes m_temp[i] − m_temp[i−1] (4th dim)
    
    phi += tau * (divergence/dx + rhodiff)  # P2, eq 4 (divides dx here)

def l2_distance(source: np.ndarray, dest: np.ndarray, dx, maxiter=100000, tau=3, mu=3e-6):
    """Compute L2 earth mover's distance between two N-dimensional arrays."""

    if len(source.shape) > MAX_DIM: # len(source.shape) gives the dimension
        raise ValueError(f"Dimensions of greater than {MAX_DIM} are not supported!")
    elif source.shape != dest.shape:
        raise ValueError(f"Dimension mismatch between source and destination! Source shape is '{source.shape}', dest shape is '{dest.shape}'.")
    rhodiff = np.array(dest-source) # difference
    phi = np.zeros_like(rhodiff) # dual function
    m = np.zeros((len(phi.shape),) + phi.shape) # tranport function; dim + phi.shape
    m_temp = np.zeros_like(m) # temp used for calculating div
    dim = len(phi.shape) # dimension
    # print(tau, mu, dx, dim)
    for i in range(maxiter): # iterate
        l2_update(phi, m, m_temp, rhodiff, tau=tau, mu=mu, dx=dx, dim=dim) # update the parameters
        if i %1000 == 0:
            print(f"Iteration: {i}, L2 distance", np.sum(np.sqrt(np.sum(m**2,axis=0))))
    return np.sum(np.sqrt(np.sum(m**2,axis=0))) # calculate minimum cost by summing up all magnitudes of m vector field


end_time = time.time()  # << END TIMER
elapsed_time = end_time - start_time
print("\nTotal runtime: {:.2f} seconds ({:.2f} minutes)".format(elapsed_time, elapsed_time / 60))