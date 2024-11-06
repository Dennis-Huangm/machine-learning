# Denis
# coding:UTF-8
import torchvision
from torchvision import transforms
from torch.utils import data
import cupy as np


def cross_entropy(y_hat_, y):  # 返回的为一个向量,输入y与y_hat都为矩阵
    return -np.log(y_hat_[range(len(y_hat_)), y]).mean()


def softmax(x):  # X是一个矩阵为经过线性层后的特征量
    X_exp = np.exp(x)
    partition = X_exp.sum(1, keepdims=True)
    return X_exp / partition


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


class Relu:
    def __init__(self):
        self.mask = None

    def __call__(self, X):
        self.mask = np.where(X > 0, np.array(1), np.array(0.01))
        return X * self.mask

    def backward(self, dout):
        return dout * self.mask


def load_data_fashion_mnist(batch_size, resize=None):
    trans = [transforms.ToTensor(), transforms.Normalize(std=0.5, mean=0.5)]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(root='./data', train=True, transform=trans, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(root='./data', train=False, transform=trans)
    return data.DataLoader(mnist_train, batch_size, shuffle=True), data.DataLoader(mnist_test, batch_size, shuffle=True)


if __name__ == '__main__':
    a = Relu()
    print(a(np.array([[1, -3, 1], [-1, -3, 1]])))
