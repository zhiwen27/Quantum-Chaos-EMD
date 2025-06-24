import numpy as np
from multiprocessing import Process, Queue
import pandas as pd
from scipy.integrate import simpson as simps
from joblib import Parallel, delayed
from scipy.interpolate import RegularGridInterpolator
import time
np.set_printoptions(threshold=np.inf)

MAX_DIM = 4
q = Queue(maxsize=1)

def wignerTimeEvol(q):
    start_time = time.time()
    t = 1
    t_f = 0.1

    theta1_min, theta1_max = 0, 2 * np.pi
    theta2_min, theta2_max = 0, 2 * np.pi
    n_theta1, n_theta2 = 40, 40

    p1_min, p1_max = -6, 6
    p2_min, p2_max = -6, 6
    n_p1, n_p2 = 40, 40

    y_min, y_max = -5, 5
    n_y = 100  # number of points for integration grid
    y1_vals = np.linspace(y_min, y_max, n_y)
    y2_vals = np.linspace(y_min, y_max, n_y)

    dtheta1 = (theta1_max - theta1_min) / (n_theta1 - 1)
    dtheta2 = (theta2_max - theta2_min) / (n_theta2 - 1)
    #dp1 = (p1_max - p1_min) / (n_p1 - 1)
    #dp2 = (p2_max - p2_min) / (n_p2 - 1)

    n_t= 100
    dt = t_f / n_t

    theta1 = np.linspace(theta1_min, theta1_max, n_theta1)
    theta2 = np.linspace(theta2_min, theta2_max, n_theta2)
    p1 = np.linspace(p1_min, p1_max, n_p1)
    p2 = np.linspace(p2_min, p2_max, n_p2)

    
    def Psi_src(theta1,theta2):
        return np.exp(- (theta1**2) - (theta2**2))
    
    def Psi_dest(theta1,theta2):
        return np.exp(- ((theta1 - 0.1) **2) - ((theta2 - 0.1 ) **2))
    
    theta1_grid, theta2_grid = np.meshgrid(theta1, theta2, indexing='ij')
    m1 = 1
    m2 = 1
    l1 = 1
    l2 = 1
    g = 1
    h_bar = 1
    c = m2 * l2 ** 2
    a = (m1 + m2) * l1 ** 2 + c
    b = m2 * l1 * l2
    d = (- h_bar ** 2 / 2) * (1 / (a * c - c ** 2 - b ** 2 * np.cos(theta2_grid)))
    big_a = c * d
    big_b = a * d + 2 * b * d * np.cos(theta2_grid)
    big_c = 2 * c * d + 2 * b * d * np.cos(theta2_grid)
    U = -m1*g*np.cos(theta1_grid)-m2*g*(l1*np.cos(theta1_grid)+l2*np.cos(theta2_grid))
    
    P_source = Psi_src(theta1_grid, theta2_grid)
    P_src = P_source / np.sqrt(np.sum(np.abs(P_source) ** 2) * dtheta1 * dtheta2)
    P_destination = Psi_dest(theta1_grid, theta2_grid)
    P_dest = P_destination / np.sqrt(np.sum(np.abs(P_destination) ** 2) * dtheta1 * dtheta2)

    def second_derivative_4th_order_parallel(P_array, axis, spacing, n_jobs=-1):
        result = np.zeros_like(P_array, dtype=complex)
        shape = P_array.shape

        def p(i):
            return i % shape[axis]

        def compute_slice(i):
            slc1 = [slice(None)] * 2
            slc2 = [slice(None)] * 2
            slc3 = [slice(None)] * 2
            slc4 = [slice(None)] * 2
            center = [slice(None)] * 2

            slc1[axis] = p(i - 2)
            slc2[axis] = p(i - 1)
            slc3[axis] = p(i + 1)
            slc4[axis] = p(i + 2)
            center[axis] = p(i)

            val = (-P_array[tuple(slc4)] + 16*P_array[tuple(slc3)] - 30*P_array[tuple(center)] +
                16*P_array[tuple(slc2)] - P_array[tuple(slc1)]) / (12 * spacing**2)
            return (tuple(center), val)

        indices = range(shape[axis])
        results = Parallel(n_jobs=n_jobs)(delayed(compute_slice)(i) for i in indices)

        for center_slice, val in results:
            result[center_slice] = val

        return result
    
    def mixed_second_derivative_4th_order_parallel(P_array, spacing1, spacing2, n_jobs=-1):
        result = np.zeros_like(P_array, dtype=complex)
        ptheta1, ptheta2 = P_array.shape

        def p1(i):
            return i % ptheta1
        
        def p2(i):
            return i % ptheta2

        def compute(i,j):
            val = (P_array[p1(i - 2), p2(j - 2)] - 8 * P_array[p1(i - 1), p2(j - 2)] + 8 * P_array[p1(i + 1), p2(j - 2)] -
                P_array[p1(i + 2), p2(j - 2)] - 8 * P_array[p1(i - 2), p2(j - 1)] + 64 * P_array[p1(i - 1), p2(j - 1)] -
                64 * P_array[p1(i + 1), p2(j - 1)] + 8 * P_array[p1(i + 2), p2(j - 1)] + 8 * P_array[p1(i - 2), p2(j + 1)] -
                64 * P_array[p1(i - 1), p2(j + 1)] + 64 * P_array[p1(i + 1), p2(j + 1)] - 8 * P_array[p1(i + 2), p2(j + 1)] -
                P_array[p1(i - 2), p2(j + 2)] + 8 * P_array[p1(i - 1), p2(j + 2)] - 8 * P_array[p1(i + 1), p2(j + 2)] +
                P_array[p1(i + 2), p2(j + 2)]) / (144 * spacing1 * spacing2)
            return (i, j, val)
        

        indices = [(i, j) for i in range(ptheta1) for j in range(ptheta2)]
        results = Parallel(n_jobs=n_jobs)(delayed(compute)(i, j) for i,j in indices)

        for i, j, val in results:
            result[i,j] = val

        return result

    def f(P_array):
        d2P_dtheta1 = second_derivative_4th_order_parallel(P_array, axis=0, spacing=dtheta1)
        d2P_dtheta2 = second_derivative_4th_order_parallel(P_array, axis=1, spacing=dtheta2)
        d2P_dtheta1_dtheta2 = mixed_second_derivative_4th_order_parallel(P_array, spacing1=dtheta1 ,spacing2=dtheta2)
        laplacian = big_a * d2P_dtheta1 + big_b * d2P_dtheta2 - big_c * d2P_dtheta1_dtheta2
        return (-1j / h_bar) * (laplacian + U) * P_array

    def compute_wigner_element(i, j, k, l, Psi):
        Y1, Y2 = np.meshgrid(y1_vals, y2_vals, indexing='ij')
        integrand_vals = np.real(
            np.conj(Psi(np.column_stack(((theta1[i] + Y1).ravel(), (theta2[j] + Y2).ravel()))).reshape(Y1.shape)) *
            Psi(np.column_stack(((theta1[i] - Y1).ravel(), (theta2[j] - Y2).ravel()))).reshape(Y1.shape) *
            np.exp(2j * (p1[k] * Y1 + p2[l] * Y2) / h_bar)
        )
        integral_y2 = simps(integrand_vals, y2_vals, axis=1)
        integral = simps(integral_y2, y1_vals)
        return i, j, k, l, integral / ((np.pi * h_bar) ** 2)

    def Wigner(P_array):
        result = np.zeros((n_theta1, n_theta2, n_p1, n_p2))
        indices = [(i, j, k, l) for i in range(n_theta1)
                                for j in range(n_theta2)
                                for k in range(n_p1)
                                for l in range(n_p2)]
        Psi = RegularGridInterpolator((theta1, theta2), P_array, bounds_error=False, fill_value=0)
        results = Parallel(n_jobs=-1, verbose=0)(
            delayed(compute_wigner_element)(i, j, k, l, Psi) for (i, j, k, l) in indices
        )
        for i, j, k, l, val in results:
            result[i, j, k, l] = val

        return result
    
    def timeEvol(P):
        for step in range(n_t):
            k1=f(P)
            P1=P+k1*dt/2
            k2=f(P1)
            P2=P+k2*dt/2
            k3=f(P2)
            P3=P+k3*dt
            k4=f(P3)
            P=P+(dt/6)*(k1+2*k2+2*k3+k4)
        return P

    source = Wigner(P_src)
    dest = Wigner(P_dest)
    q.put((0,source.copy(),dest.copy()))

    for i in range (int(t/t_f)):
        P_src = timeEvol(P_src)
        P_dest = timeEvol(P_dest)
        source = Wigner(P_src)
        dest = Wigner(P_dest)
        q.put(((i + 1) * t_f,source.copy(),dest.copy()))
    q.put(None)

    end_time = time.time()
    print(end_time-start_time)


def emdCal(q):
    import cupy as cp
    start_time = time.time()
    N = 40
    spacing = np.linspace(-6, 6, N)
    dx = spacing[1]-spacing[0]
    tau = 3
    mu = 1./(16*tau*(N-1)**2)
    def l2_update(phi: cp.ndarray, m: cp.ndarray, m_temp: cp.ndarray, rhodiff: cp.ndarray, tau, mu, dx, dim):
        m_temp[:] = -m
        m[0, :-1] += mu * (phi[1:] - phi[:-1]) / dx
        if dim > 1:
            m[1, :, :-1] += mu * (phi[:, 1:] - phi[:, :-1]) / dx
        if dim > 2:
            m[2, :, :, :-1] += mu * (phi[:, :, 1:] - phi[:, :, :-1]) / dx
        if dim > 3:
            m[3, :, :, :, :-1] += mu * (phi[:, :, :, 1:] - phi[:, :, :, :-1]) / dx
            
        norm = cp.sqrt(cp.sum(m**2, axis=0))
        shrink_factor = 1 - mu / cp.maximum(norm, mu)
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
        rhodiff = cp.array(dest-source)
        phi = cp.zeros_like(rhodiff)
        m = cp.zeros((len(phi.shape),) + phi.shape)
        m_temp = cp.zeros_like(m)
        dim = len(phi.shape)
        for i in range(maxiter):
            l2_update(phi, m, m_temp, rhodiff, tau=tau, mu=mu, dx=dx, dim=dim)
            #if i %1000 == 0:
            #    print(f"Iteration: {i}, L2 distance", cp.sum(cp.sqrt(cp.sum(m**2,axis=0))))
        return cp.sum(cp.sqrt(cp.sum(m**2,axis=0)))

    while(True):
        i = q.get()
        if i == None:
            break
        t, source, dest = i
        source /= source.sum()
        dest /= dest.sum()
        computedDistance = l2_distance(source, dest, maxiter=100000, dx=dx, tau=tau, mu = mu)
        print("Earth Mover's Distance at t = " + f"{t:.1f}" + "s:", computedDistance)
    end_time = time.time()
    print(end_time-start_time)


start_time = time.time()
if __name__ == "__main__":
    p1 = Process(target = wignerTimeEvol, args = (q,))
    p2 = Process(target = emdCal, args = (q,))
    p1.start()
    p2.start()
    p1.join()
    p2.join()
    end_time = time.time()
    print(end_time-start_time)

'''
module load cuda/12.6
export LD_LIBRARY_PATH=$CUDA_SPACK_ROOT/lib64:$LD_LIBRARY_PATH
source ~/myenv/bin/activate
'''