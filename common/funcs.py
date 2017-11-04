import numpy as np
from scipy.stats import norm


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def mean_squared_error(y, t):
    return 0.5 * np.sum((y-t)**2)


def cal_gaussian(x):
    return norm.pdf(x)
