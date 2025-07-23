import math
import numpy as np
from multiprocessing import Process, Queue
import pandas as pd
from scipy.integrate import simpson as simps
from joblib import Parallel, delayed
from scipy.interpolate import RegularGridInterpolator
import time
from scipy.interpolate import RectBivariateSpline
np.set_printoptions(threshold=np.inf)

MAX_DIM = 4
q = Queue(maxsize=1)

def wignerTimeEvol(q):
    start_time = time.time()
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

    t = 1
    t_f = 0.1
    n_t = 100
    dt = t_f / n_t
    n_t=math.ceil(t_f/dt)

    o1 = np.linspace(o1_min, o1_max, n_o1)
    o2 = np.linspace(o2_min, o2_max, n_o2)

    D = (-h_bar**2/ 2)*(1/((a*c)-(c**2)-(b**2)*(np.cos(o2)**2)))
    A = c*D
    B = a*D + 2*b*D*np.cos(o2)
    C = 2*c*D + 2*b*D*np.cos(o2)


    def Psi_src(O1,O2,f1,f2,u1,u2):
        G = (1/(np.sqrt(2*np.pi*f1*f2))) * np.exp(-((O1-u1)**2/2*f1**2)-((O2-u2)**2/2*f2**2))
        return G
    
    O1, O2 = np.meshgrid(o1, o2, indexing='ij')  # full 2D grid versions of x and p  #Grid for X,P
    P_src = Psi_src(O1,O2,f1,f2,u1,u2).astype(complex)
    f1=0.00012
    f2=0.12

    def Psi_dest(O1,O2,f1,f2,u1,u2):
        G = (1/(np.sqrt(2*np.pi*f1*f2))) * np.exp(-((O1-u1)**2/2*f1**2)-((O2-u2)**2/2*f2**2))
        return G
    
    P_dest = Psi_dest(O1,O2,f1,f2,u1,u2).astype(complex)
    U = -m1*g*l1*np.cos(O1) - m2*g*(l1*np.cos(O1) + l2*np.cos(O2))

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

        indices = range(2, shape[axis] - 2)
        results = Parallel(n_jobs=n_jobs)(delayed(compute_slice)(i) for i in indices)

        for center_slice, val in results:
            result[center_slice] = val

        return result


    def mixed_second_derivative_4th_order_parallel(P_array, dx1, dx2, n_jobs=-1):
        result = np.zeros_like(P_array, dtype=complex)
        shape = P_array.shape

        def p1(i):
            return i % P_array.shape[0]
        
        def p2(i):
            return i % P_array.shape[1]

        def compute_point(i,j):
            val = (P_array[p1(i - 2), p2(j - 2)] - 8 * P_array[p1(i - 1), p2(j - 2)] + 8 * P_array[p1(i + 1), p2(j - 2)] -
                P_array[p1(i + 2), p2(j - 2)] - 8 * P_array[p1(i - 2), p2(j - 1)] + 64 * P_array[p1(i - 1), p2(j - 1)] -
                64 * P_array[p1(i + 1), p2(j - 1)] + 8 * P_array[p1(i + 2), p2(j - 1)] + 8 * P_array[p1(i - 2), p2(j + 1)] -
                64 * P_array[p1(i - 1), p2(j + 1)] + 64 * P_array[p1(i + 1), p2(j + 1)] - 8 * P_array[p1(i + 2), p2(j + 1)] -
                P_array[p1(i - 2), p2(j + 2)] + 8 * P_array[p1(i - 1), p2(j + 2)] - 8 * P_array[p1(i + 1), p2(j + 2)] +
                P_array[p1(i + 2), p2(j + 2)])
            return (i, j, val / (144 * dx1 * dx2))

        # Only compute for valid interior points
        indices = [(i, j) for i in range(2, shape[0] - 2) for j in range(2, shape[1] - 2)]

        results = Parallel(n_jobs=n_jobs)(delayed(compute_point)(i, j) for i, j in indices)

        for i, j, val in results:
            result[i, j] = val

        return result

    def f(P_array):
        d2P_dO1 = second_derivative_4th_order_parallel(P_array, axis=0, spacing=dO1)
        d2P_dO2 = second_derivative_4th_order_parallel(P_array, axis=1, spacing=dO2)
        d2P_dO1_dO2 = mixed_second_derivative_4th_order_parallel(P_array, dO1, dO2)

        H = A*d2P_dO1 + B*d2P_dO2 - C*d2P_dO1_dO2 + U
        return (H*P_array/(1j*h_bar))
    
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

    def Wigner(P, o1, o2, n1, n2, y1_vals, y2_vals, h_bar=1):
        t0 = time.time()
        result = np.zeros((len(o1), len(o2), len(n1), len(n2)), dtype=complex)

        interp = RectBivariateSpline(o1, o2, P)

        def compute_element(i, j, k, l):
            o1_val = o1[i]
            o2_val = o2[j]
            n1_val = n1[k]
            n2_val = n2[l]

            Y1, Y2 = np.meshgrid(y1_vals, y2_vals, indexing='ij')

            psi_plus = interp(o1_val + Y1, o2_val + Y2, grid=False)
            psi_minus = interp(o1_val - Y1, o2_val - Y2, grid=False)
            integrand = np.conj(psi_plus) * psi_minus * np.exp(2j*(n1_val*Y1 + n2_val*Y2))

            integral_y2 = simps(integrand, y2_vals, axis=1)
            integral = simps(integral_y2, y1_vals)

            norm = 1 / (np.pi * h_bar)**2
            return (i, j, k, l, norm * integral)

        indices = [(i,j,k,l) for i in range(len(o1))
                            for j in range(len(o2))
                            for k in range(len(n1))
                            for l in range(len(n2))]

        results = Parallel(n_jobs=-1, verbose=0)(
            delayed(compute_element)(*idx) for idx in indices
        )

        for i, j, k, l, val in results:
            result[i, j, k, l] = val

        return result.real
    source = Wigner(P_src, o1, o2, n1, n2, y1_vals, y2_vals, h_bar=1)
    dest = Wigner(P_dest, o1, o2, n1, n2, y1_vals, y2_vals, h_bar=1)
    q.put((0,source.copy(),dest.copy()))

    for i in range (int(t/t_f)):
        P_src = timeEvol(P_src)
        P_dest = timeEvol(P_dest)
        source = Wigner(P_src, o1, o2, n1, n2, y1_vals, y2_vals, h_bar)
        dest = Wigner(P_dest, o1, o2, n1, n2, y1_vals, y2_vals, h_bar)
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