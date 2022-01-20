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
        self.dims[0] += 1
        self.weights = []
        self.lr = 0.01

        self.activations = [ac.relu for _ in range(len(dims) - 2)] + [ac.iden]

        self.loss = losses.MSE

        for n, m in zip(self.dims, self.dims[1:]):
            self.weights.append(np.random.random([m, n]) * 2 - 1)

    def forward(self, x):
        x = [1, *x]
        for i, W in enumerate(self.weights):
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

        dydw = uvT(self.activations[-1].dif(self.weights[-1] @ x), x)
        self.weights[-1] += self.lr * self.loss.dif(yhat, y) * dydw


# %%
nn = NN(2, 1)

# %%


def yf1(x): return -2 + 3 * x[0] + x[1]
def yf2(x): return 1 + x[0] - x[1]


# %%
for i in range(100):
    x = np.random.randint(-10, 10, 2)
    nn.backward(x, np.array(yf1(x)))

print(nn.weights)

# %%
nn.weights
