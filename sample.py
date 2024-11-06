# Denis
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt


def cost_gradient(W, X, Y, n):
    pre = np.dot(X, W)  # 前向传播
    G = np.dot(X.T, pre - Y) / n  # X.shape->(90,4)  故X.T.shape->(4,90)  Y.shape->(90,1) 计算梯度
    j = np.dot((pre - Y).T, pre - Y) / (2 * n)  # 计算均方误差函数的值
    # loss=(WX-Y)**2/n
    return (j, G)


def gradientDescent(W, X, Y, lr, iterations):
    n = np.size(Y)
    J = np.zeros([iterations, 1])
    for i in range(iterations):
        J[i], G = cost_gradient(W, X, Y, n)
        W = W - G * lr
    return (W, J)


plt.figure()
iterations = 20
num_sample = 50
lr = 0.0001
data = np.loadtxt('LR.txt', delimiter=',')
X, Y = data[:, 0].reshape(-1, 1), data[:, 1].reshape(-1, 1)
plt.scatter(X, Y, color='red')

n = num_sample
W = np.ones([2, 1])
X_ = np.c_[X, np.ones([n, 1])]

(W, J) = gradientDescent(W, X_, Y, lr, iterations)

plt.plot(X, np.dot(X_, W))
plt.show()
