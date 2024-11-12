# Denis
# coding:UTF-8
import torchvision
from torchvision import transforms
from torch.utils import data
import cupy as np
import random
from matplotlib import pyplot as plt

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


class Datasets:
    def __init__(self, batch_size):
        dataset = np.load('data.npy')
        self.images = dataset[:, :-1].reshape(dataset.shape[0], 20, 20)
        self.labels = dataset[:, -1].astype(np.int32)
        self.batch_size = batch_size

        self.num_samples = len(dataset)
        self.indices = list(range(self.num_samples))
        random.shuffle(self.indices)

    def __iter__(self):
        self.i = 0
        return self

    def __next__(self):
        batch_indices = self.indices[self.i: min(self.i + self.batch_size, self.num_samples)]
        self.i += len(batch_indices)
        if self.i >= self.num_samples:
            raise StopIteration
        return self.images[batch_indices], self.labels[batch_indices]


if __name__ == '__main__':
    a = Relu()
    print(a(np.array([[1, -3, 1], [-1, -3, 1]])))
    data_loader = Datasets(batch_size=32)
    for x, y in data_loader:
        print(data_loader.i)

