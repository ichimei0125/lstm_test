from myLSTMCell import MyLSTMCell
from common.gradient import numerical_gradient
import numpy as np
import matplotlib.pyplot as plt
from common.funcs import *
from test_data import test_x_real, test_y_real, test_x_train, test_y_train, func

# init parameters
train_x = test_x_train
train_t = test_y_train

learning_rate = 0.01

# init lstm cell
lstm = MyLSTMCell()
lstm1 = MyLSTMCell()

# training process
out = []
i = 0

for i in range(len(train_x) - 1):
    while True:
        c, h = lstm.cell(train_x[i])
        c1, h1 = lstm1.cell(train_x[i+1], c, h)

        dW = lstm.numerical_gradient(h, 0, 0, test_y_train[i])  # 勾配降下法

        for param in ('W_f', 'b_f', 'W_i', 'b_i', 'W_c', 'b_c', 'W_o', 'b_o'):
            lstm.params[param] -= learning_rate * dW[param]

        loss = lstm.loss(train_x[i], c, h, train_t[i])

        dW = lstm1.numerical_gradient(h1, c, h, test_y_train[i+1])  # 勾配降下法

        for param in ('W_f', 'b_f', 'W_i', 'b_i', 'W_c', 'b_c', 'W_o', 'b_o'):
            lstm1.params[param] -= learning_rate * dW[param]

        loss1 = lstm1.loss(train_x[i+1], c1, h1, train_t[i+1])

        if i == len(train_x) - 2:
            _, res = lstm.cell(test_x_train[i+1])
            out.append(res)

        if loss < 0.05 and loss1 < 0.05:
            _, res = lstm.cell(test_x_train[i])
            out.append(res)

            break

# draw picture
plt.plot(test_x_real, test_y_real)
plt.plot(test_x_train, test_y_train, 'ro')

plt.plot(train_x, out, '^')
plt.show()
