from timeEvol_EMD_4D import timeEvol
from timeEvol_EMD_4D import t_f
import numpy as np
import time
np.set_printoptions(threshold=np.inf)
MAX_DIM = 4
t = 0.1

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
        if i %1000 == 0:
            print(f"Iteration: {i}, L2 distance", np.sum(np.sqrt(np.sum(m**2,axis=0))))
    return np.sum(np.sqrt(np.sum(m**2,axis=0)))

start_time = time.time()
if __name__ == "__main__":
    N = 32 
    spacing = np.linspace(-10, 10, N)

    # read in source and dest from txt files
    source = np.loadtxt
    dest = np.loadtxt
    dx = spacing[1]-spacing[0]
    tau = 3
    mu = 1./(16*tau*(N-1)**2)
    source /= source.sum()
    dest /= dest.sum()
    computedDistance = l2_distance(source, dest, maxiter=40000, dx=dx, tau=tau, mu = mu)
    print("Earth Mover's Distance at t = 0s:", computedDistance)

    for i in range (int(t/t_f)):
        source = timeEvol(source)
        dest = timeEvol(dest)

        source /= source.sum()
        dest /= dest.sum()

        computedDistance = l2_distance(source, dest, maxiter=40000, dx=dx, tau=tau, mu = mu)
        print("Earth Mover's Distance at t = " + f"{(i + 1) * t_f:.1f}" + "s:", computedDistance)

    end_time = time.time()
    print(end_time-start_time)