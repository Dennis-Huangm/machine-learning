# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt


def read_data(addr):
    data = np.loadtxt(addr, delimiter=',')
    n = data.shape[0]
    X = np.concatenate(
        [np.ones([n, 1]), data[:, 0:1] ** 4, data[:, 1:2] ** 2, data[:, 2:3] ** 3, data[:, 3:4] ** 1, data[:, 4:5] ** 5,
         data[:, 5:6] ** 6], axis=1)

    Y = None
    if "train" in addr:
        Y = np.expand_dims(data[:, 6], axis=1)

    return (X, Y, n)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def cross_entropy(y_pre, y):
    return (-y * np.log(y_pre) - (1 - y) * np.log(1 - y_pre)).mean()


def cost_gradient(W, X, Y, n):
    pre = sigmoid(np.dot(X, W))
    G = np.dot(X.T, pre - Y) / n
    j = cross_entropy(pre, Y)
    return (j, G)


def train(W, X, Y, lr, n, iterations):
    num_fold = 10
    num_per_fold = n // num_fold

    J = np.zeros([iterations, 1])
    E_trn = np.zeros([iterations, 1])
    E_val = np.zeros([iterations, 1])

    for epoch in range(iterations):
        for i in range(num_fold):
            X_trn = np.r_[X[:i * num_per_fold], X[(i + 1) * num_per_fold:]]
            Y_trn = np.r_[Y[:i * num_per_fold], Y[(i + 1) * num_per_fold:]]
            X_val = X[i * num_per_fold:(i + 1) * num_per_fold]
            Y_val = Y[i * num_per_fold:(i + 1) * num_per_fold]
            (j, G) = cost_gradient(W, X_trn, Y_trn, n)
            J[epoch] += j
            W = W - lr * G
            E_trn[epoch] += error(W, X_trn, Y_trn)
            E_val[epoch] += error(W, X_val, Y_val)

        J[epoch] /= num_fold
        E_trn[epoch] /= num_fold
        E_val[epoch] /= num_fold

    print(E_val[-1])
    return (W, J, E_trn, E_val)


def error(W, X, Y):
    Y_hat = 1 / (1 + np.exp(-X @ W))
    Y_hat[Y_hat < 0.5] = 0
    Y_hat[Y_hat > 0.5] = 1
    return (1 - np.mean(np.equal(Y_hat, Y)))


def predict(W):
    (X, _, _) = read_data("test_data.csv")
    Y_hat = 1 / (1 + np.exp(-X @ W))
    Y_hat[Y_hat < 0.5] = 0
    Y_hat[Y_hat > 0.5] = 1

    idx = np.expand_dims(np.arange(1, 201), axis=1)
    np.savetxt("predict.csv", np.concatenate([idx, Y_hat], axis=1), header="Index,ID", comments='', delimiter=',')


iterations = 2000
lr = 0.05
(X, Y, n) = read_data("train.csv")
W = np.random.random([X.shape[1], 1])
(W, J, E_trn, E_val) = train(W, X, Y, lr, n, iterations)

plt.figure()
plt.plot(range(iterations), J)
plt.figure()
plt.ylim(0, 0.5)
plt.plot(range(iterations), E_trn, "b")
plt.plot(range(iterations), E_val, "r")
plt.show()
predict(W)
