import random

import numpy as np
from matplotlib import pyplot as plt


class Graphic:
    # y = wx + b

    def __init__(self, w, b, start_x, finish_x, noise_level):
        self.w = w
        self.b = b
        dist = finish_x - start_x
        temp = np.random.rand(50, 1) * dist + start_x
        self.points = np.array([np.array([x, w * x + b + noise_level * random.uniform(-1, 1)]) for x in temp])

    def draw(self):
        x = self.points[:, 0]
        y = self.points[:, 1]
        plt.plot(x, y, 'o')
        plt.show()


class F(Graphic):
    def __init__(self):
        super().__init__(1, 2, -1, 1, 0.1)
