import numpy as np
import time
np.set_printoptions(threshold=np.inf)
MAX_DIM = 4

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
    x,y,z,w = np.meshgrid(spacing, spacing, spacing, spacing)

    sigma = 1
    mu = np.array([1., -1., 3., 5**.5])
    source = np.exp(-.5*(x**2+y**2+z**2+w**2)/sigma)
    dest = np.exp(-.5*((x-mu[0])**2+(y-mu[1])**2+(z-mu[2])**2+(w-mu[3])**2)/sigma)
    dx = spacing[1]-spacing[0]
    tau = 3
    mu = 1./(16*tau*(N-1)**2)
    source /= source.sum()
    dest /= dest.sum()
    l2_distance(source, dest, maxiter=40000, dx=dx, tau=tau, mu = mu)
    end_time = time.time()
    print(end_time-start_time)