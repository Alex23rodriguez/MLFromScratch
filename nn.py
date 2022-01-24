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

        self.activations = [ac.relu for _ in range(len(dims) - 2)] + [ac.iden]

        self.loss = losses.MSE

        for n, m in zip(self.dims, self.dims[1:]):
            self.weights.append(np.random.random([m, n + 1]) * 2 - 1)

    def forward(self, x):
        for i, W in enumerate(self.weights):
            x = [1, *x]
            x = self.activations[i](W @ x)
        return x

    def set_activation(self, i, func):
        self.activations[i] = func

    def backward(self, x, y):
        # if j outputs:
        # dE / dwij = d(loss_j)/dwij

        # in delta rule:
        # https://en.wikipedia.org/wiki/Delta_rule
        # deltawji = lr * (tj - yj) * g'(hj)*xi, hj = sum(xi*wji), g(hj) = yj
        yhat = self.forward(x)
        x = [1, *x]

        h = self.weights[-1] @ x
        g = self.activations[-1].dif(h)

        deltaW = self.lr * uvT(self.loss.dif(yhat, y) * g, x)
        self.weights[-1] += deltaW


# %%
nn = NN(2, 1, 3)

# %%


def yf1(x): return -2 + 3 * x[0] + x[1]
def yf2(x): return 1 + x[0] - x[1]


# %%
for i in range(100):
    x = np.random.randint(-10, 10, 2)
    nn.backward(x, np.array([yf1(x), yf2(x)]))

print(nn.weights)

# %%
nn.weights

# %%
nn.weights

# %%
nn.forward([0, 0])
# %%
nn.activations
# %%
