import numpy as np
import matplotlib.pyplot as plt
from common.funcs import sigmoid

x = np.arange(-15, 15, 0.1)
y = np.tanh(x)
y1 = sigmoid(x)
y2 = y * y1

plt.plot(x, y)
plt.plot(x, y1)
plt.plot(x, y2)
plt.show()
