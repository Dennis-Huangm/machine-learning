# Denis
# coding:UTF-8
# from sklearn.datasets import load_breast_cancer
import numpy as np
from matplotlib import pyplot as plt


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def cross_entropy(y_pre, y):
    return (-y * np.log(y_pre) - (1 - y) * np.log(1 - y_pre)).mean()


# def test():
#     predict = sigmoid(np.dot(x, w)).reshape(-1)
#     for k in range(len(predict)):
#         if predict[k] >= 0.5:
#             predict[k] = 1
#         else:
#             predict[k] = 0
#     res = predict == y.reshape(-1)
#     print('accuracy:' + str(res.sum() / len(y.reshape(-1))))

def error(W, X, Y):
    Y_hat = np.sign(X @ W)
    Y_hat = np.maximum(Y_hat, 0)
    return 1 - np.mean(np.equal(Y_hat, Y))


if __name__ == '__main__':
    # cancer = load_breast_cancer()
    data = np.loadtxt('LR1.txt', delimiter=',')
    x, y = data[:, :-1], data[:, -1].reshape(-1, 1)

    num_samples, epochs = len(x), 200  # 30个特征
    w = np.random.normal(0, 0.01, size=(3, 1))
    x = np.c_[x, np.ones([num_samples, 1])]
    J = []

    for i in range(epochs):  # 全局梯度下降
        y_hat = sigmoid(np.dot(x, w))
        J.append(cross_entropy(y_hat, y))
        grad = np.dot(x.T, y_hat - y) / num_samples
        w -= 0.01 * grad

    plt.figure()
    plt.plot(range(epochs), J)
    plt.ylabel("loss")
    plt.xlabel('epoch')
    print('loss:' + str(J[-1]))
    print('error:' + str(error(w, x, y)))

    idx0 = (data[:, 2] == 0)
    idx1 = (data[:, 2] == 1)
    plt.figure()
    plt.ylim(-12, 12)
    plt.plot(data[idx0, 0], data[idx0, 1], 'go')
    plt.plot(data[idx1, 0], data[idx1, 1], 'rx')
    x1 = np.arange(-10, 10, 0.2)
    y1 = (w[2] + w[0] * x1) / -w[1]
    plt.plot(x1, y1)

    # test()
    plt.show()
