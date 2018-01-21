import math
import matplotlib.pyplot as plt
import numpy as np

plt.ion()
fig, ((a,b), (c, d)) = plt.subplots(2,2)
x = np.arange(0, 2 * np.pi, 0.1)
a.plot(x, np.sin(x))
b.plot(x, np.cos(x))
c.plot(x, np.tan(x))
d.plot(x, np.tanh(x))