import numpy as np
from common.funcs import *
from common.gradient import numerical_gradient


class MyLSTMCell:
    def __init__(self, weight_init_std=0.001):
        self.params = {'W_f': weight_init_std * np.random.randn(1, 2),
                       'b_f': np.zeros(2),
                       'W_i': weight_init_std * np.random.randn(1, 2),
                       'b_i': np.zeros(2),
                       'W_c': weight_init_std * np.random.randn(1, 2),
                       'b_c': np.zeros(2),
                       'W_o': weight_init_std * np.random.randn(1, 2),
                       'b_o': np.zeros(2)}

    def cell(self, input_x, old_c=0, old_h=0):
        # forget gate layer
        w_f, b_f = self.params['W_f'], self.params['b_f']
        f_t = sigmoid(np.sum(np.dot(w_f, [old_h, input_x]) + b_f))

        # input gate layer
        w_i, b_i = self.params['W_i'], self.params['b_i']
        i_t = sigmoid(np.sum(np.dot(w_i, [old_h, input_x]) + b_i))

        # new candidate values
        w_c, b_c = self.params['W_c'], self.params['b_c']
        _update_c = np.tanh(np.sum(np.dot(w_c, [old_h, input_x]) + b_c))

        # update state
        update_c = f_t * old_c + i_t * _update_c

        # decides what parts of the cell state going to output
        w_o, b_o = self.params['W_o'], self.params['b_o']
        o_t = sigmoid(np.sum(np.dot(w_o, [old_h, input_x]) + b_o))

        # output
        update_h = o_t * np.tanh(update_c)

        return update_c, update_h

    def loss(self, x, c, h, t):
        _, y = self.cell(x, c, h)
        loss = mean_squared_error(y, t)

        return loss

    def numerical_gradient(self, x, c, h, t):
        loss_w = lambda w: self.loss(x, c, h, t)
        grads = {'W_f': numerical_gradient(loss_w, self.params['W_f']),
                 'b_f': numerical_gradient(loss_w, self.params['b_f']),
                 'W_i': numerical_gradient(loss_w, self.params['W_i']),
                 'b_i': numerical_gradient(loss_w, self.params['b_i']),
                 'W_c': numerical_gradient(loss_w, self.params['W_c']),
                 'b_c': numerical_gradient(loss_w, self.params['b_c']),
                 'W_o': numerical_gradient(loss_w, self.params['W_i']),
                 'b_o': numerical_gradient(loss_w, self.params['b_o'])}
        return grads
