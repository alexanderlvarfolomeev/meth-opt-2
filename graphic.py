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
        return np.sum(np.dot(self.a, np.concatenate((x, np.array([1])))))


class Graphic:

    def __init__(self, linear: Linear, start: float, finish: float, noise_level: float, count: int):
        self.linear = linear
        dist = finish - start
        temp = np.random.rand(count, linear.n) * dist + start
        self.points_x = temp
        self.points_y = np.array(
            [linear(x) + noise_level * random.uniform(-1, 1)
             for x in temp])

    def draw(self):
        assert self.linear.n == 1
        x = self.points_x[:, 0]
        y = self.points_y
        plt.plot(x, y, 'o')
        plt.show()


class F(Graphic):
    def __init__(self):
        super().__init__(Linear(np.array([1, 2])), -1, 1, 0.01, 50)


class G(Graphic):
    def __init__(self):
        super().__init__(Linear(np.array([3.5, 6])), -10, 10, 1.5, 50)
