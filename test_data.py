import numpy as np
import matplotlib.pyplot as plt


def func(x):
    return 0.0004 * x ** 4

test_x_real = np.arange(-5, 5, 0.1)
test_y_real = 0.0004 * test_x_real ** 4
test_x_train = np.array([-3, -2.3, -1.7, -1.24, 0.1, 0.7, 1.3, 2.1, 3.4])
test_y_train = func(test_x_train)

# print(test_y_train)

# train_x_1 = np.array([-1.24, 0.1, 1.3, 2, 3.4])
# train_t = np.array([0.0615, 0.0004, 0.067, 0.16, 0.462])

# plt.plot(test_x_real, test_y_real)
# plt.plot(test_x_train, test_y_train, 'ro')
# plt.plot(train_x, train_t, 'bo')
# plt.show()
