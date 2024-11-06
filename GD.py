import cupy as np
import matplotlib.pyplot as plt
from tqdm import trange

x = np.arange(-3, 3, 0.3)
y = np.arange(-3, 3, 0.3)
x, y = np.meshgrid(x, y)
levels = 24

z = 3 * (1 - x) ** 2 * np.exp(-x ** 2 - (y + 1) ** 2) - 10 * (x / 5 - x ** 3 - y ** 5) * np.exp(
    -x ** 2 - y ** 2) - 1 / 3 * np.exp(-(x + 1) ** 2 - y ** 2)

fig = plt.figure(figsize=(8, 5))
plt.tick_params(labelsize=18)
plt.xlabel("$x$", fontsize=24)
plt.ylabel("$y$", fontsize=24)

plt.contourf(x.get(), y.get(), z.get(), levels=levels, cmap="rainbow")
line = plt.contour(x.get(), y.get(), z.get(), levels=levels, colors="k")

x = -0.15
y = 1.2

iterations = 100
lr = 0.143
beta1 = 0.9  # Adam参数，控制动量的衰减率
beta2 = 0.99  # Adam参数，控制梯度平方的衰减率
epsilon = 1e-8  # 避免除以零的小常数

v_x = 0
s_x = 0
v_y = 0
s_y = 0

for i in trange(iterations):
    pdx = (-6 * x ** 3 + 12 * x ** 2 - 6) * np.exp(-x ** 2 - (y + 1) ** 2) - (
            20 * x * y ** 5 + 20 * x ** 4 - 34 * x ** 2 + 2) * np.exp(-x ** 2 - y ** 2) + 2 / 3 * (
                  x + 1) * np.exp(-(x + 1) ** 2 - y ** 2)
    pdy = ((-6 * x ** 2 + 12 * x - 6) * y - 6 * x ** 2 + 12 * x - 6) * np.exp(-x ** 2 - (y + 1) ** 2) - (
            20 * y ** 6 - 50 * y ** 4 + 20 * x ** 3 * y - 4 * x * y) * np.exp(-x ** 2 - y ** 2) + 2 / 3 * y * np.exp(
        -(x + 1) ** 2 - y ** 2)

    # v_x = beta1 * v_x + (1 - beta1) * pdx
    # s_x = beta2 * s_x + (1 - beta2) * pdx ** 2
    # v_x_hat = v_x / (1 - beta1 ** (i + 1))
    # s_x_hat = s_x / (1 - beta2 ** (i + 1))
    #
    # v_y = beta1 * v_y + (1 - beta1) * pdy
    # s_y = beta2 * s_y + (1 - beta2) * pdy ** 2
    # v_y_hat = v_y / (1 - beta1 ** (i + 1))
    # s_y_hat = s_y / (1 - beta2 ** (i + 1))
    # dx = (-lr * v_x_hat / (np.sqrt(s_x_hat) + epsilon)).get()
    # dy = (-lr * v_y_hat / (np.sqrt(s_y_hat) + epsilon)).get()

    v_x = beta1 * v_x + (1 - beta1) * pdx
    v_x_hat = v_x

    v_y = beta1 * v_y + (1 - beta1) * pdy
    v_y_hat = v_y

    dx = (-lr * v_x_hat).get()
    dy = (-lr * v_y_hat).get()
    plt.arrow(x, y, dx, dy, length_includes_head=False, head_width=0.1, fc='r', ec='k')
    x += dx
    y += dy

plt.show()
