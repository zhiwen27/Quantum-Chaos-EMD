import numpy as np
from multiprocessing import Process, Queue
import pandas as pd
from scipy.integrate import simpson as simps
from joblib import Parallel, delayed
import time
np.set_printoptions(threshold=np.inf)

MAX_DIM = 4
q = Queue(maxsize=5)

def wignerTimeEvol(q):
    m=1
    w=1
    h_bar=1

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
    t = 0.1

    x1 = np.linspace(x1_min, x1_max, n_x1)
    x2 = np.linspace(x2_min, x2_max, n_x2)
    p1 = np.linspace(p1_min, p1_max, n_p1)
    p2 = np.linspace(p2_min, p2_max, n_p2)

    def Psi_src(x1,x2):
        Psi_0_0 = ((m * w / (np.pi * h_bar))**0.5)*(np.exp(-m * w*(x1**2+x2**2)/ (2 * h_bar)))
        Psi_0_1 = (np.sqrt(2/np.pi)* (m*w/(h_bar))**(3/4)) *x2 *(np.exp(-m * w * (x1**2 + x2**2) / (2 * h_bar)))
        result=np.sqrt(3/5)*Psi_0_0 + np.sqrt(2/5)*Psi_0_1
        return result
    
    def Psi_dest(x1,x2):
        Psi_0_0 = ((m * w / (np.pi * h_bar))**0.5)*(np.exp(-m * w*(x1**2+x2**2)/ (2 * h_bar)))
        Psi_0_1 = (np.sqrt(2/np.pi)* (m*w/(h_bar))**(3/4)) *x2 *(np.exp(-m * w * (x1**2 + x2**2) / (2 * h_bar)))
        result=np.sqrt(3.1/5)*Psi_0_0 + np.sqrt(1.9/5)*Psi_0_1
        return result

    Psi_src_star = lambda x1, x2: np.conj(Psi_src(x1, x2))
    Psi_dest_star = lambda x1, x2: np.conj(Psi_dest(x1, x2))

    y_min, y_max = -5, 5
    n_y = 70
    y1_vals = np.linspace(y_min, y_max, n_y)
    y2_vals = np.linspace(y_min, y_max, n_y)

    def compute_wigner_element(i, j, k, l,Psi,Psi_star):
        Y1, Y2 = np.meshgrid(y1_vals, y2_vals, indexing='ij')
        integrand_vals = np.real(
            Psi_star(x1[i] + Y1, x2[j] + Y2) *
            Psi(x1[i] - Y1, x2[j] - Y2) *
            np.exp(2j * (p1[k] * Y1 + p2[l] * Y2) / h_bar)
        )
        integral_y2 = simps(integrand_vals, x = y2_vals, axis=1)
        integral = simps(integral_y2, x = y1_vals)
        return i, j, k, l, integral / ((np.pi * h_bar) ** 2)

    def Wigner(Psi_func,Psi_func_star):
        result = np.zeros((n_x1, n_x2, n_p1, n_p2))
        indices = [(i, j, k, l) for i in range(n_x1)
                                for j in range(n_x2)
                                for k in range(n_p1)
                                for l in range(n_p2)]

        results = Parallel(n_jobs=-1, verbose=0)(
            delayed(compute_wigner_element)(i, j, k, l,Psi_func,Psi_func_star) for (i, j, k, l) in indices
        )

        for i, j, k, l, val in results:
            result[i, j, k, l] = val

        return result

    source = Wigner(Psi_src,Psi_src_star)
    dest = Wigner(Psi_dest,Psi_dest_star)

    X1,X2,P1,P2 = np.meshgrid(x1,x2,p1,p2,indexing='ij')
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

    for i in range (int(t/t_f)):
        source = timeEvol(source)
        dest = timeEvol(dest)
        q.put(((i + 1) * t_f,source.copy(),dest.copy()))
    q.put(None)

def emdCal(q):
    import cupy as cp
    N = 50
    spacing = np.linspace(-10, 10, N)
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