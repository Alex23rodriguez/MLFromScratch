# %%mbda x: int(x > 0)
import numpy as np
import activation as ac
import losses

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
        yhat = self.forward(x)
        x = [1, *x]

        dydw = self.activations[-1].dif(self.weights[-1] @ x) * x
        dw = self.lr * self.loss.dif(yhat, y) * dydw
        print(dw)


# %%
nn = NN(2, 2)

# %%

x = np.array([1, 5, 9, -2, 3, 0, 0, -2])
nn.forward(x)

# %%
nn.loss(np.array([1, 3]), np.array([4, 2]))
# %%
x
# %%

# y = 3x1 + x2 - 2

x1 = np.array([1, 2])
y1 = np.array([3])

x2 = np.array([2, 1])
y2 = np.array([5])
# %%
def yf1(x): return -2 + 3 * x[0] + x[1]
def yf2(x): return 1 + x[0] - x[1]


# %%
for i in range(100):
    x = np.random.randint(-10, 10, 2)
    nn.backward(x, np.array(yf1(x), yf2(x)))

print(nn.weights)

# %%
nn.weights
# %%
nn.forward([3, 6])
# %%
np.array([1, 2]) + np.array([2, 2])
# %%
x = np.random.randint(-10, 10, 2)
# %%
nn.backward(x, np.array([yf1(x), yf2(x)]))

# %%
x
# %%
np.array([yf1(x), yf2(x)])

# %%
