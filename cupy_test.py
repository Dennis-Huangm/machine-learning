import time
import numpy as np
import cupy as cp


def test_dot_time(is_gpu, num, dim=1024, times=10):
    # 用来统计每次计算消耗的时间
    consum_time = []
    # 使用cupy来运算矩阵的点积
    if is_gpu:
        matrix = cp
    # 使用numpy来运算矩阵的点积
    else:
        matrix = np
    # 测试10次,取平均值
    for i in range(times):
        start_time = time.time()
        # 初始化一个num×dim的二维矩阵
        a = matrix.random.normal(size=num * dim)
        a = a.reshape((num, dim))
        # 初始化一个dim×1的二维矩阵
        b = matrix.random.normal(size=dim * 1)
        # 矩阵的点积
        c = matrix.dot(a, b)
        print(c.argmax())
        end_time = time.time()
        consum_time.append(end_time - start_time)
    print("a time consume %.2f sec" % np.mean(consum_time))


if __name__ == '__main__':
    num = 1000
    test_dot_time(True, num)
    # a time consume 0.18 sec
    test_dot_time(False, num)
    # a time consume 5.48 sec
    b=cp.arange(16).reshape(4,4)
    print(cp.argmax(b,0))