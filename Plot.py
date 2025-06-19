import matplotlib.pyplot as plt
import numpy as np
emd = [0.015491063470172964,0.01549813550044657,0.015503913287070976]
times = np.arange(0, 0.3, 0.1)
plt.figure()
plt.plot(times, emd, marker='o')
plt.xlabel('Time (s)')
plt.ylabel("Earth Mover's Distance")
plt.title("EMD evolution over time")
plt.grid(True)
plt.show()