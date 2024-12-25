# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange


def update(W, X, Y, n, lr, iterations):
    J = np.zeros([iterations, 1])

    for i in trange(iterations):
        y_hat = np.dot(X, W) * Y
        temp = 0
        for j in range(n):
            if y_hat[j][0] < 0:
                W = W - lr * (-Y[j] * X[j].reshape(-1, 1))
                temp += -y_hat[j][0]

        J[i] = temp

    return (W, J)


data = np.loadtxt('Perceptron.txt', delimiter=',')

n = data.shape[0]
W = np.random.random([3, 1])
X = np.concatenate([np.ones([n, 1]), data[:, 0:2]], axis=1)
Y = np.expand_dims(data[:, 2], axis=1)

iterations = 100
lr = 0.001

(W, J) = update(W, X, Y, n, lr, iterations)

# Draw figure
idx0 = (data[:, 2] == -1)
idx1 = (data[:, 2] == 1)

plt.figure()
plt.ylim(-12, 12)
plt.plot(data[idx0, 0], data[idx0, 1], 'go')
plt.plot(data[idx1, 0], data[idx1, 1], 'rx')

x1 = np.arange(-10, 10, 0.2)
y1 = W[0] + W[1] * x1 / -W[2]
plt.plot(x1, y1)

plt.figure()
plt.plot(range(iterations), J)
plt.show()
