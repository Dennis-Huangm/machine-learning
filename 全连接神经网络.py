# Denis
# coding:UTF-8
import numpy as np
import torch
import torchvision
from torchvision import transforms
from torch.utils import data
import os
from tqdm import trange
from matplotlib import pyplot as plt

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# x学习率！！！不要老是往高的调！！！！！！！


def cross_entropy(y_hat_, y):  # 返回的为一个向量,输入y与y_hat都为矩阵
    return -np.log(y_hat_[range(len(y_hat_)), y]).mean()


def softmax(x):  # X是一个矩阵为经过线性层后的特征量
    X_exp = np.exp(x)
    partition = X_exp.sum(1, keepdims=True)
    return X_exp / partition


def load_data_fashion_mnist(batch_size, resize=None):
    trans = [transforms.ToTensor(), transforms.Normalize(std=0.5, mean=0.5)]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(root='./data', train=True, transform=trans, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(root='./data', train=False, transform=trans)
    return data.DataLoader(mnist_train, batch_size, shuffle=True), data.DataLoader(mnist_test, batch_size, shuffle=True)


def init_layers(layers_dim, num_layers):
    weight_layers = [0]
    for i in range(num_layers):
        W = np.random.normal(0, 0.1, size=(layers_dim[i] + 1, layers_dim[i + 1])).astype(np.float32)
        weight_layers.append(W)
    return weight_layers


def increase_dim(X):
    if isinstance(X, torch.Tensor):
        X = X.reshape(-1, num_features).numpy().astype(np.float32)
    X = np.c_[X, np.ones([X.shape[0], 1])]
    return X


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def forward(a, weight_layers):
    a = increase_dim(a)
    A = [a]
    for i_ in range(1, num_layers):
        a = increase_dim(sigmoid(np.dot(a, weight_layers[i_])))
        A.append(a)
    return A, softmax(np.dot(a, weight_layers[-1]))


def evaluate_accuracy(data_iter):
    acc, time = 0, 0
    for x, y in data_iter:
        A, y_hat = forward(x, weight_layers)
        cmp = y_hat.argmax(axis=1) == y.numpy()
        acc += cmp.sum() / len(cmp)
        time += 1
    return acc / time


if __name__ == '__main__':
    batch_size, num_classes = 256, 10
    train_iter, val_iter = load_data_fashion_mnist(batch_size)
    num_features, epochs = 784, 200

    layers_dim = [num_features, 256, 128, 128, 64, 32, num_classes]
    num_layers = len(layers_dim) - 1
    weight_layers = init_layers(layers_dim, num_layers)

    J = []
    accuracy = []
    plt.figure()

    pbar = trange(epochs, desc="Training", unit="epoch")
    for epoch in pbar:
        loss, turn = 0, 0
        for X, y in train_iter:
            A, y_hat = forward(X, weight_layers)
            loss += cross_entropy(y_hat, y)
            turn += 1
            # 反向传播
            dZ = y_hat - np.eye(num_classes)[y]
            for i in range(num_layers, 0, -1):
                grad_w = np.dot(A[i - 1].T, dZ) / batch_size
                weight_layers[i] -= 0.1 * grad_w  # 梯度下降
                if i > 1:
                    dZ = A[i - 1][:, :-1] * (1 - A[i - 1][:, :-1]) * np.dot(dZ, weight_layers[i][:-1, :].T)
        accuracy.append(evaluate_accuracy(val_iter))
        J.append(loss / turn)
        pbar.set_description('loss:{0:.3f}  accuracy:{1:.3f}'.format(J[-1], accuracy[-1]))

        plt.clf(), plt.xlabel('epoch')
        plt.plot(range(epoch + 1), J, label="loss", color="red")
        plt.plot(range(epoch + 1), accuracy, label="accuracy", color='blue')
        plt.legend(), plt.grid()
        plt.pause(0.01)

    print('loss:' + str(J[-1]))
    print('acc:' + str(accuracy[-1]))
    plt.show()
