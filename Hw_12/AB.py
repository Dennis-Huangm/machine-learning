# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt


def find_best_split(X_i, Y, weight):
    b = [None] * (X_i.shape[0] - 1)
    sign = [None] * (X_i.shape[0] - 1)
    e = [None] * (X_i.shape[0] - 1)
    Y_hat = np.zeros_like(Y)

    X_sorted = np.sort(X_i)
    for i in range(X_i.shape[0] - 1):
        b[i] = (X_sorted[i] + X_sorted[i + 1]) / 2
        idx = (X_i > b[i])  # 找出处于这条线右边的点
        keys, counts = np.unique(Y[idx], return_counts=True)  # Y[idx]会保留在x=b右侧的样本，根据这些样本的实际类别来确定现在这个线性模型划分的情况
        sign[i] = keys[np.argmax(counts)]  # 即究竟是右边为1，还是左边为1
        Y_hat[idx] = sign[i]
        Y_hat[~idx] = -sign[i]
        e[i] = np.sum(weight[Y_hat != Y])  # 错误率

    idx = np.argmin(e)
    return b[idx], sign[idx], e[idx]


def train_weak_learner(X, Y, weight):
    b = [None] * X.shape[1]
    sign = [None] * X.shape[1]
    e = [None] * X.shape[1]

    for i in range(X.shape[1]):
        b[i], sign[i], e[i] = find_best_split(X[:, i], Y, weight)

    idx = np.argmin(e)
    W = np.zeros([X.shape[1], 1])
    W[idx] = sign[idx]  # W是用来确定是基于x1还是x2
    b = -sign[idx] * b[idx]  # 考虑进了直线x=b左侧为+1类，右侧反而为-1类的情况

    return W, b, np.min(e)


def ada_boost(X, Y, n, T):
    weight = np.full([n, 1], 1 / n)

    alpha = [None] * T
    W = [None] * T
    b = [None] * T
    ###### Calculate all alpha_i, W_i and b_i
    for t in range(T):
        W[t], b[t], e = train_weak_learner(X, Y, weight)  # 在weight改变的情况下后面训练出的模型自然有所不同
        alpha[t] = 0.5 * np.log10((1 - e) / e)
        weights = weight * np.e**(-alpha[t] * Y * (X @ W[t] + b[t]))
        weight = weights / weights.sum()
    return alpha, W, b


def error(alphas, Ws, bs, X, Y):
    z = 0
    for alpha, W, b in zip(alphas, Ws, bs):
        z += alpha * np.sign(X @ W + b)
    Y_hat = np.sign(z)

    return (1 - np.mean(np.equal(Y_hat, Y)))


data = np.loadtxt('AB1.txt', delimiter=',')

n = data.shape[0]
X = data[:, 0:2]
Y = np.expand_dims(data[:, 2], axis=1)

T = 5  ###### Determine how many weak learners do you need
alphas, Ws, bs = ada_boost(X, Y, n, T)
print(error(alphas, Ws, bs, X, Y))

# Draw figure
idx0 = (data[:, 2] == -1)
idx1 = (data[:, 2] == 1)

plt.figure()
plt.ylim(-10, 10)
plt.plot(data[idx0, 0], data[idx0, 1], 'bx')
plt.plot(data[idx1, 0], data[idx1, 1], 'ro')

x1 = np.arange(-10, 10, 0.1)
for W, b in zip(Ws, bs):
    y1 = (b + W[0] * x1) / (-W[1] + 1e-8)
    plt.plot(x1, y1)
plt.show()
