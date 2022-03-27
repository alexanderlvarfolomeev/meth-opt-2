import random

import numpy as np
from matplotlib import pyplot as plt


class Linear:
    # y = wx + b
    def __init__(self, w: float, b: float):
        self.w = w
        self.b = b

    def __call__(self, x: float) -> float:
        return self.w * x + self.b


class Graphic:

    def __init__(self, linear: Linear, start_x: float, finish_x: float, noise_level: float):
        self.linear = linear
        dist = finish_x - start_x
        temp = np.random.rand(50, 1) * dist + start_x
        self.points = np.array(
            [np.array([x, linear.w * x + linear.b + noise_level * random.uniform(-1, 1)])
             for x in temp])

    def draw(self):
        x = self.points[:, 0]
        y = self.points[:, 1]
        plt.plot(x, y, 'o')
        plt.show()


class F(Graphic):
    def __init__(self):
        super().__init__(Linear(1, 2), -1, 1, 0.1)


class G(Graphic):
    def __init__(self):
        super().__init__(Linear(3.5, 6), -10, 10, 0.15)
