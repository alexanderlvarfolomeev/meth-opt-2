import random

import numpy as np
from matplotlib import pyplot as plt
from numpy import ndarray


class Linear:
    # y = a_0 * x_0 + a_1 * x_1 + ... + a_{n - 1} * x_{n - 1} + a_{n} * 1
    def __init__(self, a: ndarray):
        self.a = a
        self.n = a.shape[0] - 1

    def __call__(self, x: ndarray) -> float:
        return np.sum(np.dot(self.a, x))


class Graphic:

    def __init__(self, linear: Linear, start: float, finish: float, noise_level: float, count: int, seed: int):
        self.linear = linear
        dist = finish - start
        np.random.seed(seed)
        temp = np.random.rand(count, linear.n) * dist + start
        self.points_x = np.array([np.concatenate((x, np.array([1]))) for x in temp])
        self.points_y = np.array(
            [linear(x) + noise_level * random.uniform(-1, 1)
             for x in self.points_x])

    def draw(self):
        assert self.linear.n == 1
        x = self.points_x[:, 0]
        y = self.points_y
        plt.plot(x, y, 'o')
        plt.show()


class F(Graphic):
    def __init__(self):
        super().__init__(Linear(np.array([1, 2])), -1, 1, 0.01, 50, 121391674)


class G(Graphic):
    def __init__(self):
        super().__init__(Linear(np.array([3.5, 6])), -10, 10, 1.5, 50, 386484234)


class F3d(Graphic):
    def __init__(self):
        super().__init__(Linear(np.array([1, 1, 2])), -1, 1, 0.01, 100, 28347834)


class F10d(Graphic):
    def __init__(self):
        super(F10d, self).__init__(Linear(
            np.array([1.1, 34, 2.1, 0.2343, 34.4, -1.3, 1, 12.2, 0.01, -0.01])), -1, 1, 0.01, 100, 364937374)
