# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt


# Utilities
def onehotEncoder(Y, ny):
    return np.eye(ny)[Y]


# Xavier Initialization
def initWeights(M):
    l = len(M)
    W = []
    B = []

    for i in range(1, l):
        # Xavier initialization
        factor = np.sqrt(6 / (M[i - 1] + M[i]))
        W.append(np.random.uniform(-factor, factor, size=(M[i - 1], M[i])))
        B.append(np.zeros([1, M[i]]))

    return W, B


# Forward propagation
def networkForward(X, W, B):
    l = len(W)
    A = [None for i in range(l + 1)]

    A[0] = X
    for i in range(1, l + 1):
        Z = np.dot(A[i - 1], W[i - 1]) + B[i - 1]
        if i < l:
            A[i] = 1 / (1 + np.exp(-Z))  # sigmoid activation
        else:
            A[i] = np.exp(Z) / np.sum(np.exp(Z), axis=1, keepdims=True)  # softmax activation

    return A


# Backward propagation
def networkBackward(Y, A, W):
    l = len(W)
    dW = [None for i in range(l)]
    dB = [None for i in range(l)]

    n = Y.shape[0]
    dZ = [None for i in range(l + 1)]

    dZ[-1] = A[-1] - Y
    for i in range(l - 1, -1, -1):
        dZ[i] = np.dot(dZ[i + 1], W[i].T) * (A[i] * (1 - A[i]))  # sigmoid activation

    for i in range(l):
        dW[i] = np.dot(A[i].T, dZ[i + 1]) / n
        dB[i] = np.sum(dZ[i + 1], axis=0, keepdims=True) / n

    return dW, dB


# Update weights by gradient descent
# def networkBackward(Y, A, W):
#     l = len(W)
#     dW = [None for i in range(l)]
#     dB = [None for i in range(l)]
#
#     n = Y.shape[0]
#     dZ = [None for i in range(l + 1)]
#
#     dZ[-1] = A[-1] - Y
#     for i in range(l - 1, -1, -1):
#         dZ[i] = np.dot(dZ[i + 1], W[i].T) * (A[i] * (1 - A[i]))  # sigmoid activation
#
#     for i in range(l):
#         dW[i] = np.dot(A[i].T, dZ[i + 1]) / n
#         dB[i] = np.sum(dZ[i + 1], axis=0, keepdims=True) / n
#
#     return dW, dB
def updateWeights(W, B, dW, dB, lr):
    l = len(W)

    for i in range(l):
        W[i] = W[i] - lr * dW[i]
        B[i] = B[i] - lr * dB[i]

    return W, B


# Compute regularized cost function
def cost(A_l, Y, W):
    n = Y.shape[0]
    c = -np.sum(Y * np.log(A_l)) / n

    return c


def train(X, Y, M, lr=0.1, iterations=3000):
    costs = []
    W, B = initWeights(M)

    for i in range(iterations):
        A = networkForward(X, W, B)
        c = cost(A[-1], Y, W)
        dW, dB = networkBackward(Y, A, W)
        W, B = updateWeights(W, B, dW, dB, lr)

        if i % 100 == 0:
            print("Cost after iteration %i: %f" % (i, c))
            costs.append(c)

    return W, B, costs


def predict(X, W, B):
    A = networkForward(X, W, B)
    Y_out = np.argmax(A[-1], axis=1)
    Y_out = onehotEncoder(Y_out, Y.shape[1])

    return Y_out


def test(Y, X, W, B):
    Y_out = predict(X, W, B)
    acc = np.sum(Y_out * Y) / Y.shape[0]
    print("Training accuracy is: %f" % (acc))

    return acc


iterations = 5000
lr = 0.1

data = np.load("data.npy")

X = data[:, :-1]
Y = data[:, -1].astype(np.int32)
(n, m) = X.shape
Y = onehotEncoder(Y, 10)

M = [400, 25, 10]  # number of neurons in each layer
W, B, costs = train(X, Y, M, lr, iterations)

plt.figure()
plt.plot(range(len(costs)), costs)
plt.show()
test(Y, X, W, B)
