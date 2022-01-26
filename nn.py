# %%mbda x: int(x > 0)
import numpy as np
import activation as ac
import losses

# %%


def uvT(u, v):
    return np.array([u]).T @ np.array([v])
# %%


class NN:
    def __init__(self, *dims):
        assert len(dims) > 1, 'not enough dimensions'
        self.dims = list(dims)
        # self.dims[0] += 1
        self.weights = []
        self.lr = 0.01

        self.activ = [ac.relu for _ in range(len(dims) - 2)] + [ac.iden]

        self.loss = losses.MSE

        for n, m in zip(self.dims, self.dims[1:]):
            self.weights.append(np.random.random([m, n + 1]) * 2 - 1)

    def forward(self, x, intermediate=False):
        A = []
        Z = []
        for i, W in enumerate(self.weights):
            A.append(np.hstack((1, x)))
            Z.append(W @ A[-1])
            x = self.activ[i](Z[-1])

        return (Z, A, x) if intermediate else x

    def set_activation(self, i, func):
        self.activ[i] = func

    def delta_rule(self, x, y):
        # if j outputs:
        # dE / dwij = d(loss_j)/dwij

        # in delta rule:
        # https://en.wikipedia.org/wiki/Delta_rule
        # deltawji = lr * (tj - yj) * g'(hj)*xi, hj = sum(xi*wji), g(hj) = yj
        yhat = self.forward(x)
        x = np.array([1, *x])

        h = self.weights[-1] @ x
        g = self.activ[-1].dif(h)

        deltaW = self.lr * uvT(self.loss.dif(yhat, y) * g, x)
        self.weights[-1] += deltaW

    def backprop(self, x, y):
        Z, A, yhat = self.forward(x, True)
        deltW = []

        delta = - self.loss.dif(yhat, y) * self.activ[-1].dif(Z[-1])

        deltW.append(self.lr * uvT(delta, A[-1]))

        for L in range(-2, -len(self.dims), -1):
            delta = self.weights[L + 1][:, 1:].T @ delta * \
                (self.activ[L].dif(Z[L]))

            deltW.append(self.lr * uvT(delta, A[L]))

        for w, dW in zip(self.weights, reversed(deltW)):
            w += dW


# %% ############## BACKPROP TEST 1
nn = NN(2, 2, 1)

nn.weights = [np.array([np.array([0.3, 0.6, -0.1]),
                        np.array([0.5, -0.3, 0.4])]),

              np.array([np.array([-0.2, 0.4, 0.1])])]

nn.set_activation(0, ac.sigmoid)
nn.set_activation(1, ac.sigmoid)

nn.lr = 0.25

x = np.array([0, 1])
y = np.array([1])
# %%
nn.forward([0, 1], True)
# %%
nn.backprop(x, y)
nn.weights

# %% ################ BACKPROP TEST 2
nn = NN(2, 2, 2)

nn.weights = [
    np.array([0.35, 0.15, 0.2, 0.35, 0.25, 0.3]).reshape(2, 3),
    np.array([0.6, 0.4, 0.45, 0.6, 0.5, 0.55]).reshape(2, 3)
]

nn.set_activation(0, ac.sigmoid)
nn.set_activation(1, ac.sigmoid)

nn.lr = 0.5

x = np.array([0.05, 0.1])
y = np.array([0.01, 0.99])
# %%
nn.forward(x, True)
# %%
nn.backprop(x, y)
nn.weights
# %% #####


def yf1(x): return -2 + 3 * x[0] + x[1]
def yf2(x): return 1 + x[0] - x[1]


# %%
for i in range(100):
    x = np.random.randint(-10, 10, 2)
    nn.backward(x, np.array([yf1(x), yf2(x)]))

print(nn.weights)
