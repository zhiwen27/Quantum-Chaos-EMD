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

def Psi_src(x):
    Psi_0 = ((m * w / (np.pi * h_bar))**0.25)*(np.exp(-m * w * x**2 / (2 * h_bar)))
    Psi_1 = ((m * w / (np.pi * h_bar))**0.25)*(np.exp(-m * w * x**2 / (2 * h_bar))*(np.sqrt(2*m*w/h_bar))*x)
    result=np.sqrt(3/5)*Psi_0 + np.sqrt(2/5)*Psi_1
    return result
Psi_star_src=Psi_src

def Psi_dest(x):
    Psi_0 = ((m * w / (np.pi * h_bar))**0.25)*(np.exp(-m * w * x**2 / (2 * h_bar)))
    Psi_1 = ((m * w / (np.pi * h_bar))**0.25)*(np.exp(-m * w * x**2 / (2 * h_bar))*(np.sqrt(2*m*w/h_bar))*x)
    result=np.sqrt(3.1/5)*Psi_0 + np.sqrt(1.9/5)*Psi_1
    return result
Psi_star_dest=Psi_dest

# can do faster integration
def Wigner_src(Psi_func):
    result=np.zeros((n_x,n_p))
    for i in range(n_x):
        for j in range(n_p):
            integrand=lambda y: np.real(Psi_star_src(x[i]+y)*(Psi_src(x[i]-y))*np.exp(2j*p[j]*y/h_bar))
            integral_value, _ = quad(integrand,-np.inf,np.inf)
            result[i,j]=integral_value/(np.pi*h_bar)
    return result
def Wigner_dest(Psi_func):
    result=np.zeros((n_x,n_p))
    for i in range(n_x):
        for j in range(n_p):
            integrand=lambda y: np.real(Psi_star_dest(x[i]+y)*(Psi_dest(x[i]-y))*np.exp(2j*p[j]*y/h_bar))
            integral_value, _ = quad(integrand,-np.inf,np.inf)
            result[i,j]=integral_value/(np.pi*h_bar)
    return result
W_src=Wigner_src(Psi_src)
W_dest = Wigner_dest(Psi_dest)


U = 0.5*m*w*w*x**2
X, P = np.meshgrid(x, p,indexing='ij')  # full 2D grid versions of x and p  #Grid for X,P


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
        W[0:2, :] = W[-2:, :] = W[:, 0:2] = W[:, -2:] = 0  # Zero the boundary
    return W

start_time = time.time()
if __name__ == "__main__":
    np.savetxt("Wigner_init_src.txt", W_src, delimiter=" ")
    np.savetxt("Wigner_init_dest.txt", W_dest, delimiter=" ")

    N = 51
    spacing = np.linspace(-6,6,N)
    
    source = np.loadtxt("Wigner_init_src.txt")
    dest = np.loadtxt("Wigner_init_dest.txt")
    dx = spacing[1]-spacing[0]
    tau = 3
    mu = 1./(16*tau*(N-1)**2)
    source /= source.sum()
    dest /= dest.sum()

    # ❗️ apply the for loop here
    # ❗️ need improvement on parameters for accuracy

    #W_src = timeEvol(W_src)
    #W_dest = timeEvol(W_dest)

    #np.savetxt("Wigner_final_src.txt", W_src, delimiter=" ")
    #np.savetxt("Wigner_final_dest.txt", W_dest, delimiter=" ")

    computedDistance = l2_distance(source, dest, maxiter=40000, dx=dx, tau=tau, mu = mu)
    print("Earth Mover's Distance at t = 0:", computedDistance)
    end_time = time.time()
    print(end_time-start_time)