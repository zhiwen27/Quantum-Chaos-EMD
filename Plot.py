import matplotlib.pyplot as plt
import numpy as np
t_f = 0.1
t = 1
emd = []
times = np.arange(0, t + t_f, t_f)
plt.figure()
plt.plot(times, emd, marker='o')
plt.xlabel('Time (s)')
plt.ylabel("Earth Mover's Distance")
plt.title("EMD evolution over time")
plt.grid(True)
plt.show()