# Denis
# -*- coding: utf-8 -*-
import numpy as np

a = np.arange(10)
b = np.flip(a)
print(b)

c = np.random.randn(10, 10)
print(np.max(c, axis=1))

print(np.where(c > 0.5, 1, 0))

print(np.mean(c, axis=1))
print(np.var(c, axis=1))

d = np.random.randn(5, 5, 3).T
e = np.random.randn(5, 5)
print((d * e).T.shape)
