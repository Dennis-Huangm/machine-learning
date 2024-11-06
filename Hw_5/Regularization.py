# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def cross_entropy(y_pre, y):
    return (-y * np.log(y_pre) - (1 - y) * np.log(1 - y_pre)).mean()


def read_data(addr):
    data = np.loadtxt(addr, delimiter=',')

    n = data.shape[0]

    ###### You may modify this section to change the model
    X = np.concatenate([np.ones([n, 1]),
                        np.expand_dims(np.power(data[:, 0], 0), axis=1),
                        np.expand_dims(np.power(data[:, 1], 0), axis=1),
                        np.expand_dims(np.power(data[:, 2], 0), axis=1),
                        np.expand_dims(np.power(data[:, 3], 0), axis=1),
                        np.expand_dims(np.power(data[:, 4], 0), axis=1),
                        np.expand_dims(np.power(data[:, 5], 0), axis=1),
                        np.expand_dims(np.power(data[:, 6], 0), axis=1),
                        np.expand_dims(np.power(data[:, 7], 0), axis=1),
                        np.expand_dims(np.power(data[:, 0], 1), axis=1),
                        np.expand_dims(np.power(data[:, 1], 1), axis=1),
                        np.expand_dims(np.power(data[:, 2], 1), axis=1),
                        np.expand_dims(np.power(data[:, 3], 1), axis=1),
                        np.expand_dims(np.power(data[:, 4], 1), axis=1),
                        np.expand_dims(np.power(data[:, 5], 1), axis=1),
                        np.expand_dims(np.power(data[:, 6], 1), axis=1),
                        np.expand_dims(np.power(data[:, 7], 1), axis=1),
                        np.expand_dims(np.power(data[:, 0], 2), axis=1),
                        np.expand_dims(np.power(data[:, 1], 2), axis=1),
                        np.expand_dims(np.power(data[:, 2], 2), axis=1),
                        np.expand_dims(np.power(data[:, 3], 2), axis=1),
                        np.expand_dims(np.power(data[:, 4], 2), axis=1),
                        np.expand_dims(np.power(data[:, 5], 2), axis=1),
                        np.expand_dims(np.power(data[:, 6], 2), axis=1),
                        np.expand_dims(np.power(data[:, 7], 2), axis=1),
                        np.expand_dims(np.power(data[:, 0], 3), axis=1),
                        np.expand_dims(np.power(data[:, 1], 3), axis=1),
                        np.expand_dims(np.power(data[:, 2], 3), axis=1),
                        np.expand_dims(np.power(data[:, 3], 3), axis=1),
                        np.expand_dims(np.power(data[:, 4], 3), axis=1),
                        np.expand_dims(np.power(data[:, 5], 3), axis=1),
                        np.expand_dims(np.power(data[:, 6], 3), axis=1),
                        np.expand_dims(np.power(data[:, 7], 3), axis=1),
                        np.expand_dims(np.power(data[:, 0], 4), axis=1),
                        np.expand_dims(np.power(data[:, 1], 4), axis=1),
                        np.expand_dims(np.power(data[:, 2], 4), axis=1),
                        np.expand_dims(np.power(data[:, 3], 4), axis=1),
                        np.expand_dims(np.power(data[:, 4], 4), axis=1),
                        np.expand_dims(np.power(data[:, 5], 4), axis=1),
                        np.expand_dims(np.power(data[:, 6], 4), axis=1),
                        np.expand_dims(np.power(data[:, 7], 4), axis=1)], axis=1)
    ###### You may modify this section to change the model

    Y = None
    if "train" in addr:
        Y = np.expand_dims(data[:, -1], axis=1)

    return (X, Y, n)


def cost_gradient(W, X, Y, n, lambd):
    pre = sigmoid(np.dot(X, W))
    G = np.dot(X.T, pre - Y) / n + lambd * np.sign(W)
    j = cross_entropy(pre, Y) + lambd * np.sum(np.abs(W))

    return (j, G)


def train(W, X, Y, lr, n, iterations, lambd):
    J = np.zeros([iterations, 1])

    for i in range(iterations):
        (J[i], G) = cost_gradient(W, X, Y, n, lambd)
        W = W - lr * G
    err = error(W, X, Y)

    return (W, J, err)


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


def ensemble_predict(w, num_learner):
    (X, _, n) = read_data("test_data.csv")
    Y_hat = ensemble(X, w, n, num_learner)
    idx = np.expand_dims(np.arange(1, 201), axis=1)
    np.savetxt("predict.csv", np.concatenate([idx, Y_hat], axis=1), header="Index,ID", comments='', delimiter=',')


def ensemble(X, w, n, num_learner):
    res = np.zeros([num_learner, n, 1])
    y_hat = np.ones([n, 1])
    for i in range(num_learner):
        res[i] = 1 / (1 + np.exp(-X @ w[i]))
        res[i][res[i] < 0.5] = 0
        res[i][res[i] > 0.5] = 1
    for i in range(n):
        results = res[:, i, 0].astype(np.int64)
        # print(results)
        counts = np.bincount(results)
        y_hat[i] = np.argmax(counts)
    if n == 400:
        print(1 - np.mean(np.equal(y_hat, Y)))
    return y_hat


iterations = 1000
# lr = 0.05
# lambd = 0

lr = [0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05, 0.055, 0.06]
lambd = [0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008, 0.0009, 0.001]

(X, Y, n) = read_data("train.csv")
W = np.random.random([X.shape[1], 1])
num_learner = 10

w = np.ones([num_learner, 41, 1])
for i in range(num_learner):
    a = np.random.randint(0, 10)
    b = np.random.randint(0, 10)
    (w[i], J, err) = train(W, X, Y, lr[a], n, iterations, lambd[b])

ensemble(X, w, n, num_learner)
ensemble_predict(w, num_learner)
# print(err)
# print(J[-1])

plt.figure()
plt.plot(range(iterations), J)
# predict(W)
plt.show()