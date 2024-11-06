# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt


def softmax(x):  # X为一个矩阵为经过线性层后的特征量
    X_exp = np.exp(x)
    partition = X_exp.sum(1, keepdims=True)
    return X_exp / partition


def cost_gradient(W, X, Y, n):
    Y_hat = softmax(np.dot(X, W))
    G = np.dot(X.T, Y_hat - Y) / n
    j = -np.log((Y_hat * Y).sum(axis=1)).sum()

    return (j, G)


def train(W, X, Y, n, lr, iterations):
    J = np.zeros([iterations, 1])

    for i in range(iterations):
        (J[i], G) = cost_gradient(W, X, Y, n)
        W = W - lr * G

    return (W, J)


def error(W, X, Y):
    Y_hat = softmax(np.dot(X, W))
    pred = np.argmax(Y_hat, axis=1)
    label = np.argmax(Y, axis=1)

    return (1 - np.mean(np.equal(pred, label)))


iterations = 5000
lr = 0.02

data = np.loadtxt('SR.txt', delimiter=',')

n = data.shape[0]
X = np.concatenate([np.ones([n, 1]),
                    np.expand_dims(data[:, 0], axis=1),
                    np.expand_dims(data[:, 1], axis=1),
                    np.expand_dims(data[:, 2], axis=1)],
                   axis=1)
Y = data[:, 3].astype(np.int32)
c = np.max(Y) + 1
Y = np.eye(c)[Y]

W = np.random.random([X.shape[1], c])

(W, J) = train(W, X, Y, n, lr, iterations)

plt.figure()
plt.plot(range(iterations), J)

print(error(W, X, Y))
plt.show()
