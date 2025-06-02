#import cupy as cp
import numpy as np
import time
import math
import numpy as np
from scipy.integrate import quad
np.set_printoptions(threshold=np.inf)
MAX_DIM = 4

def l2_update(phi: np.ndarray, m: np.ndarray, m_temp: np.ndarray, rhodiff: np.ndarray, tau, mu, dx, dim):
    """Do an L2 update."""

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
    """Compute L2 earth mover's distance between two N-dimensional arrays."""

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

m=1
w=1
h_bar=1

# Define the domain boundaries and number of boxes
x_min = -6
x_max = 6
n_x = 51   # Number of boxes (points = boxes)
p_min = -6
p_max = 6
n_p = 51

t_f=0.1 #final time
n_t=1000     #Number of steps taken for time
dt=t_f/n_t

# Compute step sizes
dx = (x_max - x_min) / (n_x - 1)
dp = (p_max - p_min) / (n_p - 1)

# Create grid points
x = x_min + dx * np.arange(n_x)
p = p_min + dp * np.arange(n_p)
#print("\nArray of all x points:\n",x)

def Psi(x):
    Psi_0 = ((m * w / (np.pi * h_bar))**0.25)*(np.exp(-m * w * x**2 / (2 * h_bar)))
    Psi_1 = ((m * w / (np.pi * h_bar))**0.25)*(np.exp(-m * w * x**2 / (2 * h_bar))*(np.sqrt(2*m*w/h_bar))*x)
    result=np.sqrt(3/5)*Psi_0 + np.sqrt(2/5)*Psi_1
    return result

# can do faster integration
def Wigner(Psi_func):
        result=np.zeros((n_x,n_p))
        for i in range(n_x):
            for j in range(n_p):
                integrand=lambda y: np.real(Psi_star(x[i]+y)*(Psi(x[i]-y))*np.exp(2j*p[j]*y/h_bar))
                integral_value, _ = quad(integrand,-np.inf,np.inf)
                result[i,j]=integral_value/(np.pi*h_bar)
        return result

def f(W_array):
        result=np.zeros((n_x,n_p))

        def xW_dev(W_array):
            result=np.zeros((n_x,n_p))
            for i in range(2,n_x-2):
                for j in range(2,n_p-2):
                    # Using 4th-order central differences
                    result[i, j] = (-W_array[i + 2, j] + 8 * W_array[i + 1, j]- 8 * W_array[i - 1, j] + W_array[i - 2, j]) / (12 * dx)
            return result
        xW_p=xW_dev(W_array)

        def pW_dev(W_array):
            result=np.zeros((n_x,n_p))
            for i in range(2,n_x-2):
                for j in range(2,n_p-2):
                    # Using 4th-order central differences
                    result[i, j] = (-W_array[i, j + 2] + 8 * W_array[i, j + 1]- 8 * W_array[i, j - 1] + W_array[i, j - 2]) / (12 * dp)
            return result
        
        pW_p=pW_dev(W_array)

        result[2:-2,2:-2] = ((-1/m)*(P[2:-2,2:-2])*(xW_p[2:-2,2:-2])) + (m*(w**2)*(X[2:-2,2:-2])*(pW_p[2:-2,2:-2]))
        return result

start_time = time.time()
if __name__ == "__main__":
    Psi_star=Psi
    W=Wigner(Psi)

    U = 0.5*m*w*w*x**2
    X, P = np.meshgrid(x, p,indexing='ij')
    
    np.savetxt("Wigner_init.txt", W, delimiter=" ")
    # have source and dest here
    N = 49
    spacing = np.linspace(-5,5,N) # space the grids with width determined by N
    x, y = np.meshgrid(spacing, spacing) # create 2D arrays x and y with spacing
    source = np.exp(-((.8-x)**2+(.8-y)**2)/2) / (2*np.pi) ** .5 # source with x and y
    dest = np.exp(-((.6-x)**2+(.6-y)**2)/2) / (2*np.pi) ** .5 # dest with x and y
    np.savetxt("source.txt", source, delimiter=" ")
    np.savetxt("dest.txt", dest, delimiter=" ")
    source = np.loadtxt("source.txt")
    dest = np.loadtxt("dest.txt")
    dx = spacing[1]-spacing[0] # spacing dx
    tau = 3 # common safe default for 2D/ 3D problems
    mu = 1./(16*tau*(N-1)**2) # ensures the operator norm of the gradient is controlled
                              # stability conditions for the primal-dual algorithm; proportioanl to (N-1)^2
    source /= source.sum() # normalization
    dest /= dest.sum() # normalization
    l2_distance(source, dest, maxiter=40000, dx=dx, tau=tau, mu = mu) # calculate distance
    end_time = time.time()
    print(end_time-start_time)