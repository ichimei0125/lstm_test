from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt

x = np.arange(-5, 5, 0.1)
x_axis = [1, 2, 3, 4, 5, 6, 7, 8]
x_dot = [1, 1, 2, 3, 5, 8, 13, 21]
x_dot_random = [2, 1, 3, 5, 13, 21, 8, 1]


def cal_gaussian(input):
    return norm.pdf(input)

plt.subplot(221)
plt.plot(x, cal_gaussian(x))

plt.subplot(222)
plt.plot(x_dot, cal_gaussian(x_dot), 'bo')
print(cal_gaussian(x_dot))

plt.subplot(223)
plt.plot(x_dot_random, cal_gaussian(x_dot_random), 'ro')
print(cal_gaussian(x_dot_random))

plt.subplot(224)
plt.plot(x_axis, cal_gaussian(x_dot), 'bo')
plt.plot(x_axis, cal_gaussian(x_dot_random), 'ro')
plt.show()
