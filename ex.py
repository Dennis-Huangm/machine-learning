# Denis
# coding:UTF-8
import torch
import cupy as np
import os


print(torch.cuda.is_available())

x = np.arange(16).reshape(2, 2, 2, 2)
img = np.pad(x, [(0, 0), (0, 0), (1, 1), (1, 1)], 'constant')
print(img)
x = np.random.normal(0, 0.01, 6)
print(x)
print(x.shape)
# modules = [Conv2d(in_channels=1, out_channels=6, padding=1), F.Relu(), Conv2d(6, 16, 1), F.Relu(),
#            Pool2d(), Conv2d(16, 32, 1), F.Relu(), Conv2d(32, 120, 1), F.Relu(), Pool2d(), Linear(7 * 7 * 120, 512),
#            F.Relu(), Linear(512, 256), F.Relu(), Linear(256, 128), F.Relu(), Linear(128, 10)]
dic = {'a': img}  # 字典为可变对象
b = dic['a']
dic['a'] += 1
print(b)


def f(arr):
    arr -= 3


f(img)
print(img)


a = torch.arange(4).reshape(2, 2)
print(a)
print(a[1,:])
