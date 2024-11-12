# Denis
# coding:UTF-8
import os
from tqdm import tqdm
import sys
import cupy as np
import random

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
"""不考虑矩形卷积核"""


def onehotEncoder(Y, ny):
    return np.eye(ny)[Y]


def cross_entropy(y_hat_, y):  # 返回的为一个向量,输入y与y_hat都为矩阵
    return -np.log(y_hat_[range(len(y_hat_)), y]).mean()


def softmax(x):  # X是一个矩阵为经过线性层后的特征量
    X_exp = np.exp(x)
    partition = X_exp.sum(1, keepdims=True)
    return X_exp / partition


class Relu:
    def __init__(self):
        self.mask = None

    def __call__(self, X):
        self.mask = np.where(X > 0, np.array(1), np.array(0.01))
        return X * self.mask

    def backward(self, dout):
        return dout * self.mask


class Dataloader:
    def __init__(self, batch_size):
        dataset = np.load('data.npy')
        self.images = np.expand_dims((dataset[:, :-1].reshape(dataset.shape[0], 20, 20).transpose(0, 2, 1)), 1)
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

    def __len__(self):
        return self.num_samples // self.batch_size


class Conv2d:
    def __init__(self, in_channels: int, out_channels: int,
                 padding=0, kernel_size=3):
        self.kernel_size = kernel_size
        self.in_c = in_channels
        self.out_c = out_channels
        self.padding = padding
        self.weights = None
        self.bias = None
        self.grad_w = None
        self.grad_b = None
        self.X = None

        self.origin_len = None
        self.out_size = None
        self.init = False

    def __call__(self, X):  # 输入X为4维[batch_size,num_channels,height,width]
        self.origin_len = X.shape[-1]
        self.out_size = int((self.origin_len - self.kernel_size + 2 * self.padding) + 1)
        if not self.init:
            self.init_weights()
            self.init = True
        self.X = np.pad(X, [(0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)], 'constant')
        outputs = np.zeros((X.shape[0], self.out_c, self.out_size, self.out_size))  # 输出也为4维
        for i in range(self.out_size):
            for j in range(self.out_size):
                outputs[:, :, i, j] = (self.X[:, None, :, i:i + self.kernel_size, j:j + self.kernel_size] *
                                       self.weights).sum(axis=(2, 3, 4))
        outputs += self.bias
        return outputs

    def init_weights(self):
        self.weights = np.random.normal(0, 0.29, size=(self.out_c, self.in_c, self.kernel_size, self.kernel_size))
        # self.weights = xavier_init(shape=(self.out_c, self.in_c, self.kernel_size, self.kernel_size))
        self.bias = np.random.normal(0, 0.01, (self.out_c, self.out_size, self.out_size))

    def backward(self, dout):
        self.grad_w = np.zeros_like(self.weights)
        grad_x = np.zeros_like(self.X)
        height = dout.shape[-1]
        for i in range(self.kernel_size):
            for j in range(self.kernel_size):
                self.grad_w[:, :, i, j] = (dout[:, :, None, :, :] * self.X[:, None, :, i:i + height,
                                                                    j:j + height]).sum(axis=(3, 4)).mean(axis=0)
        for i in range(height):
            for j in range(height):
                grad_x[:, :, i:i + self.kernel_size, j:j + self.kernel_size] += (
                        dout[:, :, i, j].reshape(-1, self.out_c, 1, 1, 1) * self.weights).sum(axis=1)
        self.grad_b = dout.mean(axis=0)
        return grad_x[:, :, self.padding:self.origin_len + self.padding, self.padding:self.origin_len + self.padding]


class Pool2d:  # 超出自动补零
    def __init__(self, stride=2, pool_size=2, mode='max'):
        self.mode = mode
        self.pool_size = pool_size
        self.stride = stride
        self.X = None
        self.maps = None

    def __call__(self, X):
        self.X = X
        len = X.shape[-1]
        scale = len / self.stride
        out_size = int(scale) + 1 if scale % 1 else int(scale)
        outputs = np.zeros((X.shape[0], X.shape[1], out_size, out_size))
        self.maps = np.zeros(X.shape)
        for i in range(out_size):
            for j in range(out_size):
                if self.mode == 'max':
                    range_input = X[:, :, self.stride * i:min(self.stride * i + self.pool_size, len),
                                  self.stride * j:min(self.stride * j + self.pool_size, len)]
                    max_value = range_input.max(axis=(2, 3))
                    outputs[:, :, i, j] = max_value
                    self.maps[:, :, self.stride * i:min(self.stride * i + self.pool_size, len),
                    self.stride * j:min(self.stride * j + self.pool_size, len)] = \
                        range_input >= max_value.reshape(X.shape[0], -1, 1, 1)
                if self.mode == 'avg':
                    outputs[:, :, i, j] = X[:, :, self.stride * i:min(self.stride * i + self.pool_size, len),
                                          self.stride * j:min(self.stride * j + self.pool_size, len)].mean(axis=(2, 3))
        return outputs

    def backward(self, dout):
        grad_x = np.zeros_like(self.X)
        length = dout.shape[-1]
        for i in range(length):
            for j in range(length):
                grad_x[:, :, self.stride * i: min(self.stride * i + self.pool_size, length),
                    self.stride * j:min(self.stride * j + self.pool_size, length)] = dout[:, :, i, j]. \
                       reshape(-1, dout.shape[1], 1, 1) * self.maps[:, :, self.stride * i: min(self.stride * i
                           + self.pool_size, length), self.stride * j:min(self.stride * j + self.pool_size, length)]
        return grad_x


class Linear:
    def __init__(self, num_hiddens, num_classes):
        self.num_hiddens = num_hiddens
        self.num_classes = num_classes
        self.weights = np.random.normal(0, 0.03, size=(num_hiddens, num_classes))
        self.bias = np.random.normal(0, 0.01, (1, num_classes))
        self.origin_Shape = None
        self.X = None
        self.grad_w = None
        self.grad_b = None

    def __call__(self, X):
        self.origin_Shape = X.shape
        X = X.reshape(X.shape[0], -1)
        self.X = X
        return np.dot(X, self.weights) + self.bias

    def backward(self, dout):
        self.grad_w = np.dot(self.X.T, dout) / self.X.shape[0]
        self.grad_b = dout.mean(axis=0, keepdims=True)
        return np.dot(dout, self.weights.T).reshape(*self.origin_Shape)


class Sequential:
    def __init__(self, *args):
        self._module = dict()
        for id_, module in enumerate(args):
            self._module[str(id_)] = module

    def __call__(self, features):
        for block in self._module.values():
            features = block(features)
        return softmax(features)

    def backward(self, dout):
        for module in reversed(list(self._module.values())):
            dout = module.backward(dout)
            if hasattr(module, 'weights'):
                # 改为Adam时可直接将权重作为参数传入，同时传入index作为module的编号
                module.weights -= module.grad_w * alpha
                module.bias -= module.grad_b * alpha


class Accumulator:
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, item):
        return self.data[item]


def xavier_init(shape, distribution='normal'):
    fan_in = shape[1] * shape[2] * shape[3]
    fan_out = shape[0] * shape[2] * shape[3]

    if distribution == 'uniform':
        limit = np.sqrt(6 / (fan_in + fan_out))
        return np.random.uniform(-limit, limit, size=shape)
    elif distribution == 'normal':
        std = np.sqrt(2 / (fan_in + fan_out))
        return np.random.normal(0, std, size=shape)
    else:
        raise ValueError("Invalid distribution type. Choose 'uniform' or 'normal'.")


def test(Y_hat, Y):
    Y_out = np.zeros_like(Y)
    idx = np.argmax(Y_hat, axis=1)
    Y_out[range(Y.shape[0]), idx] = 1
    accuracy = np.sum(Y_out * Y) / Y.shape[0]
    return accuracy


if __name__ == '__main__':
    batch_size, num_classes = 32, 10
    train_iter = Dataloader(batch_size)
    epochs, alpha = 50, 0.01
    net = Sequential(Conv2d(in_channels=1, out_channels=32, padding=1), Relu(), Pool2d(),
               Conv2d(in_channels=32, out_channels=64, padding=1), Relu(), Pool2d(), Linear(5 * 5 * 64, num_classes))
    metric = Accumulator(3)

    for t in range(epochs):
        pbar = tqdm(train_iter, ascii=True, unit="batch", file=sys.stdout)
        for X, y in pbar:
            y_hat = net(X)
            loss = cross_entropy(y_hat, y)
            acc = test(y_hat, onehotEncoder(y, 10))
            # 反向传播
            dout = y_hat - np.eye(num_classes)[y]  # softmax层
            net.backward(dout)
            pbar.set_description('Training: epoch {0}/{1}  loss:{2:.3f}'.format(t + 1, epochs, loss))
            metric.add(loss, acc, 1)
        print(f"============= Training Accuracy:{metric[1] / metric[2]} Loss:{metric[0] / metric[2]}=============\n")
        metric.reset()
