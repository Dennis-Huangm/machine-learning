# Denis
# coding:UTF-8
import os
from tqdm import tqdm
import functional as F
from torch.utils.dlpack import to_dlpack
import sys
import cupy as np

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
"""不考虑矩形卷积核"""


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
        self.weights = np.random.normal(0, 0.29, size=(self.out_c, self.in_c, self.kernel_size,
                                                       self.kernel_size)).astype(np.float32)
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
        self.weights = np.random.normal(0, 0.03, size=(num_hiddens, num_classes)).astype(np.float32)
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


def forward(features, modules):
    # features = features.numpy().astype(np.float32)
    features = np.from_dlpack(to_dlpack(features.to("cuda")))
    for module in modules:
        features = module(features)
    return F.softmax(features)


def evaluate_accuracy(data_iter, modules):
    accuracy, time = 0, 0
    for x, y in tqdm(data_iter, desc='Testing', ascii=True, unit="batch", file=sys.stdout):
        y_hat = forward(x, modules)
        cmp = np.asnumpy(y_hat).argmax(axis=1) == y.numpy()
        accuracy += cmp.sum() / len(cmp)
        time += 1
    return accuracy / time


if __name__ == '__main__':
    batch_size, num_classes = 32, 10

    # epochs = 50
    # modules = [Conv2d(in_channels=1, out_channels=4), F.Relu(), Conv2d(in_channels=4, out_channels=8), F.Relu(),
    #           Pool2d(), Conv2d(8, 32), F.Relu(), Conv2d(32, 64), F.Relu(), Pool2d(), Linear(4 * 4 * 64, num_classes)]
    # # 重写nn.Sequential，即可将面向过程改成面向对象，将forward方法封装进Sequential类的__call__方法中
    # alpha = 0.001
    #
    # for t in range(epochs):
    #     cost, iteration = 0, 0
    #     pbar = tqdm(train_iter, ascii=True, unit="batch", file=sys.stdout)
    #     for X, y in pbar:
    #         y_hat = forward(X, modules)
    #         loss = F.cross_entropy(y_hat, y)
    #         cost += loss
    #         iteration += 1
    #         # 反向传播
    #         dout = y_hat - np.eye(num_classes)[y]
    #         for index, module in enumerate(reversed(modules)):
    #             dout = module.backward(dout)
    #             if type(module) == Linear or type(module) == Conv2d:
    #                 # 改为Adam时可直接将权重作为参数传入，同时传入index作为module的编号
    #                 module.weights -= module.grad_w * alpha
    #                 module.bias -= module.grad_b * alpha
    #         pbar.set_description('Training: epoch {0}/{1}  loss:{2:.3f}'.format(t + 1, epochs, loss))
    #     acc = evaluate_accuracy(val_iter, modules)
    #     print(f"=============== Test Accuracy:{acc} Loss:{cost / iteration}===============\n")
