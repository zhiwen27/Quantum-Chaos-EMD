#import cupy as cp
import numpy as np
import time
np.set_printoptions(threshold=np.inf)
MAX_DIM = 4

# Performs one iteration of updating the dual (phi) and transport (m) variables to minimize the cost associated with the L2 EMD.
# phi: dual potential function (same shape as input); m: transport vector field
# m_temp: compute the divergence; rhodiff: difference between the dest and source
# tau: step size for gradient; mu: regularization parameter; dx: grid spacing; dim: dimensionality
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

start_time = time.time()
if __name__ == "__main__":
    N = 32 
    spacing = np.linspace(-10, 10, N)
    x,y,z,w = np.meshgrid(spacing, spacing, spacing, spacing)

    sigma = 1
    mu = np.array([1., -1., 3., 5**.5])
    source = np.exp(-.5*(x**2+y**2+z**2+w**2)/sigma)
    dest = np.exp(-.5*((x-mu[0])**2+(y-mu[1])**2+(z-mu[2])**2+(w-mu[3])**2)/sigma)
    #N = 50
    #spacing = np.linspace(-5,5,N) # space the grids with width determined by N
    #x, y = np.meshgrid(spacing, spacing) # create 2D arrays x and y with spacing
    #source = np.exp(-((.8-x)**2+(.8-y)**2)/2) / (2*np.pi) ** .5 # source with x and y
    #dest = np.exp(-((.6-x)**2+(.6-y)**2)/2) / (2*np.pi) ** .5 # dest with x and y
    #rho = 0.65
    #p = 0.25
    #theta = np.pi / 5
    #source_exp = x **2 + (y **2 / rho **2)
    #source = np.exp(-source_exp)
    #dest_exp = x **2 * (np.cos(theta) **2 / p **2 + p **2 * np.sin(theta) **2 / rho **2) + 2*x*y*np.sin(theta)*np.cos(theta)*(1/ p **2 - p **2 / rho **2) + y **2 * (np.sin(theta)**2/ p **2 + p **2 * np.cos(theta) **2 / rho **2)
    #dest = np.exp(-dest_exp)
    dx = spacing[1]-spacing[0] # spacing dx
    tau = 3 # common safe default for 2D/ 3D problems
    mu = 1./(16*tau*(N-1)**2) # ensures the operator norm of the gradient is controlled
                              # stability conditions for the primal-dual algorithm; proportioanl to (N-1)^2
    source /= source.sum() # normalization
    dest /= dest.sum() # normalization
    l2_distance(source, dest, maxiter=40000, dx=dx, tau=tau, mu = mu) # calculate distance
    end_time = time.time()
    print(end_time-start_time)

'''
m: a vector field showing how much mass moves in each direction and where; the sum of magnitudes of m corresponds to how far
and how much mass has to move.

l2_update:
gradient step: m += mu * ∇ϕ. Move m in the direction where the dual cost increases the most; satisfying the divergence constraint
∇⋅m = rho_dest - rho_source = rhodiff
shrink factor: proximal operator, enforces the regularization condition to tell strong transport directions.

The algorithm updates the transport field m by taking a gradient step — using the gradient of the dual potential ϕ, which indicates
how the potential changes most rapidly — and then applies a shrink factor that acts as a regularization to control the magnitude of 
m. This process guides m toward representing the optimal transport directions and intensities (i.e., where and how much mass should
move). Finally, the minimal transport cost is computed as the sum of the magnitudes of the vectors in m, since this sum corresponds
to the total "effort" required to move the mass from the source to the destination.
'''

'''
module load cuda/12.6
export LD_LIBRARY_PATH=$CUDA_SPACK_ROOT/lib64:$LD_LIBRARY_PATH
source ~/myenv/bin/activate
'''