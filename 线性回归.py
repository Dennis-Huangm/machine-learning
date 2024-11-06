# Denis
# coding:UTF-8
import numpy as np
import matplotlib.pyplot as plt


# 线性回归是一种更一般的概念，而最小二乘法是实现线性回归的一种具体方法。

def generate_dataset(w, b, num_sample):  # y=wx+b+杂音
    X = np.random.randn(num_sample)  # 随机初始化
    Y = X * w + b
    Y += np.random.normal(0, 0.5, X.shape)  # 噪音
    return X.reshape(-1, 1), Y.reshape(-1, 1)


if __name__ == '__main__':
    data = np.loadtxt('LR.txt', delimiter=',')
    x, y = data[:, 0].reshape(-1, 1), data[:, 1].reshape(-1, 1)
    w = np.zeros([2, 1])  # 对w与b做初始化，以一个向量的形式来进行存储
    num_samples, epochs = x.shape[0], 20
    # x, y = generate_dataset(w=2.3, b=1.2, num_sample=num_samples)
    features = np.c_[x, np.ones([num_samples, 1])]  # 样本值以一个向量的形式来进行存储，
    J = [0] * epochs  # 储存每次迭代的损失函数

    plt.figure()
    for i in range(epochs):
        y_pre = np.dot(features, w)
        J[i] = np.dot((y_pre - y).T, y_pre - y).reshape(-1) / (2 * num_samples)  # 计算损失
        grad = np.dot(features.T, y_pre - y) / num_samples  # 计算梯度
        '''均方误差代价损失函数为(y_pre-y)^2/2n,其中y_pre=wx
        #故对w求导后得到的梯度应为(y_pre-y)x/n  n为样本数量'''
        w -= 0.0001 * grad  # 梯度下降，更新参数
        plt.clf()
        plt.scatter(x, y, color='red', label='sample')
        plt.plot(x, np.dot(features, w), label='fit')
        plt.legend(), plt.pause(0.01)
    print('loss:' + str(J[-1]))
    plt.figure()
    plt.plot(list(range(epochs)), J)
    plt.ylabel("loss")
    plt.xlabel('epoch')
    plt.show()
