# Denis
# coding:UTF-8
import numpy as np
import torchvision
from torchvision import transforms
from torch.utils import data
import os
from tqdm import trange
from matplotlib import pyplot as plt

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def load_data_fashion_mnist(batch_size, resize=None):
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(root='./data', train=True, transform=trans, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(root='./data', train=False, transform=trans)
    return data.DataLoader(mnist_train, batch_size, shuffle=True), data.DataLoader(mnist_test, batch_size, shuffle=True)


def cross_entropy(y_hat, y):  # 返回的为一个向量,输入y与y_hat都为矩阵
    return -np.log(y_hat[range(len(y_hat)), y]).mean()


def softmax(x):  # X为一个矩阵为经过线性层后的特征量
    X_exp = np.exp(x)
    partition = X_exp.sum(1, keepdims=True)
    return X_exp / partition


def evaluate_accuracy(data_iter):
    acc, time = 0, 0
    for X_, y in data_iter:
        tmp_ = X_.reshape(-1, num_features)
        X = np.c_[tmp_, np.ones([tmp_.shape[0], 1])]
        y_hat = softmax(np.dot(X, W)).argmax(axis=1)
        cmp = y_hat == y.numpy()
        acc += cmp.sum() / len(cmp)
        time += 1
    return acc / time


if __name__ == '__main__':
    batch_size = 256
    train_iter, val_iter = load_data_fashion_mnist(batch_size)
    num_features, epochs = 784, 10
    num_class = 10
    W = np.random.normal(0, 0.01, size=(784 + 1, 10))  # 784为图片的像素总数即特征数，多出来的1为b偏置项

    J = []
    accuracy = []
    for i in trange(epochs, desc="Training", unit="epoch"):
        loss, turn = 0, 0
        for X_, y in train_iter:
            tmp = X_.reshape(-1, num_features)  # 应该存在内存泄漏
            X = np.c_[tmp, np.ones([tmp.shape[0], 1])]  # 应该存在内存泄漏
            y_hat = softmax(np.dot(X, W))
            grad = np.dot(X.T, y_hat - np.eye(num_class)[y]) / batch_size  # 独热编码
            W -= 0.1 * grad
            loss += cross_entropy(y_hat, y)
            turn += 1
        accuracy.append(evaluate_accuracy(val_iter))
        J.append(loss / turn)
        # print(J[-1])

    plt.figure()
    plt.plot(range(epochs), J, label="loss", color="red")
    plt.plot(range(epochs), accuracy, label="accuracy", color='blue')
    plt.xlabel('epoch')
    plt.legend()
    plt.grid()
    print('loss:' + str(J[-1]))
    print('acc:' + str(accuracy[-1]))
    plt.show()
