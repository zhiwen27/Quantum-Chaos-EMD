import math
import numpy as np
from scipy.integrate import simpson as simps
from joblib import Parallel, delayed
from tqdm import tqdm
import os
import multiprocessing
import time
import threading
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import RegularGridInterpolator


start_time = time.time()

# ========== TIMER FUNCTION ==========
def show_timer():
    while not timer_stop[0]:
        elapsed = time.time() - start_time
        print(f"⏱️ {int(elapsed)} seconds elapsed...", flush=True)
        time.sleep(120)

timer_stop = [False]
timer_thread = threading.Thread(target=show_timer)
timer_thread.start()

# ========== SYSTEM CHECK ==========
print("🧠 CPUs visible to Python:", multiprocessing.cpu_count())
print("🔧 CPUs allocated by SLURM:", os.environ.get("SLURM_CPUS_ON_NODE"))

# ========== PARAMETERS ==========
m = 1
w = 1
h_bar = 1

n_x1 = n_x2 = n_p1 = n_p2 = 50
n_y = 100

x1 = np.linspace(-7, 7, n_x1)
x2 = np.linspace(-7, 7, n_x2)
p1 = np.linspace(-7, 7, n_p1)
p2 = np.linspace(-7, 7, n_p2)
y1 = np.linspace(-7, 7, n_y)
y2 = np.linspace(-7, 7, n_y)

Y1, Y2 = np.meshgrid(y1, y2, indexing='ij')

# ========== WAVEFUNCTION ==========
def Psi(x1, x2):
    Psi_0_0 = ((m * w / (np.pi * h_bar))**0.5) * np.exp(-m * w * (x1**2 + x2**2) / (2 * h_bar))
    Psi_0_1 = (np.sqrt(2/np.pi)*(m**3 * w**3 / (h_bar**3))**0.25) * x2 * np.exp(-m * w * (x1**2 + x2**2) / (2 * h_bar))
    return np.sqrt(3/5) * Psi_0_0 + np.sqrt(2/5) * Psi_0_1

Psi_star = lambda x1, x2: np.conj(Psi(x1, x2))

# ========== WIGNER ELEMENT FUNCTION ==========
def compute_wigner_element(i, j, k, l):
    x1_val = x1[i]
    x2_val = x2[j]
    p1_val = p1[k]
    p2_val = p2[l]

    integrand_vals = np.real(
        Psi_star(x1_val + Y1, x2_val + Y2) *
        Psi(x1_val - Y1, x2_val - Y2) *
        np.exp(2j * (p1_val * Y1 + p2_val * Y2) / h_bar)
    )

    integral_y2 = simps(integrand_vals, y2, axis=1)
    integral = simps(integral_y2, y1)
    return i, j, k, l, integral / ((np.pi * h_bar) ** 2)

# ========== INDEX SETUP ==========
indices = [(i, j, k, l)
        for i in range(n_x1)
        for j in range(n_x2)
        for k in range(n_p1)
        for l in range(n_p2)]

# ========== PARALLEL COMPUTATION ==========
print(f"\n🚀 Starting Wigner 4D computation on {len(indices):,} points...")

results = Parallel(n_jobs=16)(
    delayed(compute_wigner_element)(i, j, k, l)
    for i, j, k, l in tqdm(indices,disable=True)
)


# ========== ASSEMBLE WIGNER ARRAY ==========
W = np.zeros((n_x1, n_x2, n_p1, n_p2))

for i, j, k, l, val in results:
    W[i, j, k, l] = val

# ========== STOP TIMER THREAD ==========
elapsed_time = time.time() - start_time
minutes = int(elapsed_time // 60)
seconds = int(elapsed_time % 60)
print(f"\n✅ Done. Total time: {minutes} min {seconds} sec ({elapsed_time:.2f} seconds)")


# ========== EXAMPLE LOOKUPS ==========
# Create interpolator from the 4D Wigner data
W_interp = RegularGridInterpolator((x1, x2, p1, p2), W, bounds_error=False, fill_value=None)

def lookup(x1_target, x2_target, p1_target, p2_target):
    point = np.array([[x1_target, x2_target, p1_target, p2_target]])
    W_val = W_interp(point)[0]
    print(f"W(x1={x1_target:.2f}, x2={x2_target:.2f}, p1={p1_target:.2f}, p2={p2_target:.2f}) = {W_val:.8f}")

print("n_x = n_p =", n_x1, ", n_y =", n_y)

# Lookup values at desired points with smooth interpolation
lookup(-1.52, -1.52, -1.52, -1.52)
lookup(-1.0, -1.0, -1.0, -1.0)
lookup(-0.5, -0.5, -0.5, -0.5)
lookup(0.5, 0.5, 0.5, 0.5)
lookup(1.0, 1.0, 1.0, 1.0)
lookup(1.5, 1.5, 1.5, 1.5)

'''
# 3D Visual
# Fix x2=0, p2=0
j_fixed = np.argmin(np.abs(x2 - 0))
l_fixed = np.argmin(np.abs(p2 - 0))

# Create meshgrid for plotting
X1, P1 = np.meshgrid(x1, p1, indexing='ij')
W_slice = W[:, j_fixed, :, l_fixed]

# 3D surface plot
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X1, P1, W_slice, cmap='plasma', edgecolor='none')

ax.set_xlabel('x1')
ax.set_ylabel('p1')
ax.set_zlabel('W(x1, p1)')
ax.set_title('3D Wigner Function Slice at x2 = 0, p2 = 0')
plt.tight_layout()
plt.show()
'''