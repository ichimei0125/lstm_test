from myLSTMCell import MyLSTMCell
from common.gradient import numerical_gradient
import numpy as np
import matplotlib.pyplot as plt
from common.funcs import *
from test_data import test_x_real, test_y_real, test_x_train, test_y_train, func

# init parameters
train_x = np.array([])
for i in range(len(test_x_train) - 4):
    train_x = np.append(train_x, test_x_train[i])

train_t = np.array([])
for i in range(len(test_y_train) - 4):
    train_t = np.append(train_t, test_y_train[i + 2])

learning_rate = 0.01

# init lstm cell
lstm = MyLSTMCell()
lstm1 = MyLSTMCell()
lstm2 = MyLSTMCell()
lstm3 = MyLSTMCell()
lstm4 = MyLSTMCell()
lstm5 = MyLSTMCell()
lstm6 = MyLSTMCell()

# training process
for j in range(100):
    c, h = lstm.cell(train_x[0])
    c1, h1 = lstm1.cell(train_x[1], c, h)
    c2, h2 = lstm2.cell(train_x[2], c1, h1)

    dW = lstm.numerical_gradient(h, 0, 0, test_y_train[0])  # 勾配降下法

    for param in ('W_f', 'b_f', 'W_i', 'b_i', 'W_c', 'b_c', 'W_o', 'b_o'):
        lstm.params[param] -= learning_rate * dW[param]

    dW = lstm1.numerical_gradient(h1, c, h, test_y_train[1])  # 勾配降下法

    for param in ('W_f', 'b_f', 'W_i', 'b_i', 'W_c', 'b_c', 'W_o', 'b_o'):
        lstm1.params[param] -= learning_rate * dW[param]

    dW = lstm2.numerical_gradient(h2, c1, h1, train_t[0])  # 勾配降下法

    for param in ('W_f', 'b_f', 'W_i', 'b_i', 'W_c', 'b_c', 'W_o', 'b_o'):
        lstm2.params[param] -= learning_rate * dW[param]

    # --
    c1, h1 = lstm1.cell(train_x[1], c, h)
    c2, h2 = lstm2.cell(train_x[2], c1, h1)
    c3, h3 = lstm3.cell(train_x[3], c2, h2)

    dW = lstm3.numerical_gradient(h3, c2, h2, train_t[1])  # 勾配降下法

    for param in ('W_f', 'b_f', 'W_i', 'b_i', 'W_c', 'b_c', 'W_o', 'b_o'):
        lstm3.params[param] -= learning_rate * dW[param]

    # --
    c2, h2 = lstm2.cell(train_x[2], c1, h1)
    c3, h3 = lstm3.cell(train_x[3], c2, h2)
    c4, h4 = lstm4.cell(train_x[4], c3, h3)

    dW = lstm4.numerical_gradient(h4, c3, h3, train_t[2])  # 勾配降下法

    for param in ('W_f', 'b_f', 'W_i', 'b_i', 'W_c', 'b_c', 'W_o', 'b_o'):
        lstm4.params[param] -= learning_rate * dW[param]

    # --
    c3, h3 = lstm3.cell(train_x[3], c2, h2)
    c4, h4 = lstm4.cell(train_x[4], c3, h3)
    c5, h5 = lstm5.cell(h4, c4, h4)

    dW = lstm5.numerical_gradient(h5, c4, h4, train_t[3])  # 勾配降下法

    for param in ('W_f', 'b_f', 'W_i', 'b_i', 'W_c', 'b_c', 'W_o', 'b_o'):
        lstm5.params[param] -= learning_rate * dW[param]

    # --
    c4, h4 = lstm4.cell(train_x[4], c3, h3)
    c5, h5 = lstm5.cell(h4, c4, h4)
    c6, h6 = lstm6.cell(h5, c5, h5)

    dW = lstm6.numerical_gradient(h6, c5, h5, train_t[4])  # 勾配降下法

    for param in ('W_f', 'b_f', 'W_i', 'b_i', 'W_c', 'b_c', 'W_o', 'b_o'):
        lstm6.params[param] -= learning_rate * dW[param]

out = []
_, res = lstm.cell(test_x_train[0])
out.append(res)
_, res = lstm1.cell(test_x_train[1], _, res)
out.append(res)
_, res = lstm2.cell(test_x_train[2], _, res)
out.append(res)
_, res = lstm3.cell(test_x_train[3], _, res)
out.append(res)
_, res = lstm4.cell(test_x_train[4], _, res)
out.append(res)
_, res = lstm5.cell(test_x_train[5], _, res)
out.append(res)
_, res = lstm6.cell(test_x_train[6], _, res)
out.append(res)

x_new = np.array([])
for i in range(len(test_x_train) - 4):
    x_new = np.append(x_new, test_x_train[i + 2])
plt.plot(x_new, train_t, 'ro')

x = np.append([test_x_train[0], test_x_train[1]], x_new)
plt.plot(x, out, '^')
print(out[0], out[1])

plt.plot(test_x_real, test_y_real)

plt.show()
