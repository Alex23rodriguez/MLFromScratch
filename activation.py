from cProfile import label
import numpy as np


def relu(x):
    return np.max([x, np.zeros(len(x))], axis=0)


relu.dif = lambda x: np.int(x > 0)


def tanh(x):
    return np.tanh(x)


tanh.dif = lambda x: 1 - tanh(x) * tanh(x)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


sigmoid.dif = lambda x: sigmoid(x) * (1 - sigmoid(x))


def iden(x):
    return x


iden.dif = lambda x: np.ones(len(x))
