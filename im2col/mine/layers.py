import numpy as np
import utils


class ConvLayer:
    def __init__(self, W, B, stride=1, padding=0):
        self.W = W
        self.B = B
        self.stride = stride
        self.padding = padding

        self.x = None
        self.col = None
        self.col_W = None

        self.dW = None
        self.db = None

    def forward(self, x):
        FN, C, FH, FW = self.W.shape
        N, C, H, W = x.shape
        out_h = int(1 + (H + 2 * self.padding - FH) / self.stride)
        out_w = int(1 + (W + 2 * self.padding - FW) / self.stride)

        col = utils.im2col(x, FH, FW, self.stride, self.padding)
        col_W = self.W.reshape(FN, -1).T
        output = np.dot(col, col_W) + self.B

        output = output.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)

        self.x = x
        self.col = col
        self.col_W = col_W

        return output


class ReluLayer:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0

        return out

    def backward(self, dOut):
        dOut[self.mask] = 0
        dx = dOut

        return dx


class MaxPoolingLayer:
    def __init__(self, height, width, stride=1, padding=0):
        self.height = height
        self.width = width
        self.stride = stride
        self.padding = padding

        self.x = None
        self.arg_max = None

    def forward(self, x):
        N, C, H, W = x.shape
        output_height = int(1 + (H - self.height) / self.stride)
        output_width = int(1 + (W - self.width) / self.stride)

        col = utils.im2col(x, self.height, self.width, self.stride, self.padding)
        col = col.reshape(-1, self.height * self.width)

        arg_max = np.argmax(col, axis=1)
        out = np.max(col, axis=1)
        out = out.reshape(N, output_height, output_width, C).transpose(0, 3, 1, 2)

        self.x = x
        self.arg_max = arg_max

        return out

    def backward(self, dOut):
        dOut = dOut.transpose(0, 2, 3, 1)

        pool_size = self.height * self.width
        dMax = np.zeros((dOut.size, pool_size))
        dMax[np.arange(self.arg_max.size), self.arg_max.flatten()] = dOut.flatten()
        dMax = dMax.reshape(dOut.shape + (pool_size,))

        dCol = dMax.reshape(dMax.shape[0] * dMax.shape[1] * dMax.shape[2], -1)

        return utils.col2im(dCol, self.x.shape, self.height, self.width, self.stride, self.padding)


class SoftmaxLayer:
    def __init__(self):
        self.loss = None
        self.y = None
        # 监督数据
        self.t = None

    def forward(self, x):
        self.y = utils.softmax(x)

        return self.y

    def backward(self, t):
        self.t = t
        batch_size = self.t.shape[0]
        self.loss = utils.cross_entropy_error_softmax(self.y, self.t)
        if self.t.size == self.y.size:
            dx = (self.y - self.t) / batch_size

        return dx