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


def normalized_points(points: ndarray):
    if (points.ndim == 2):
        for row in range(points.shape[1] - 1):
            min_row = np.min(points[:, row])
            max_row = np.max(points[:, row])
            min_max_diff = max_row - min_row
            if min_max_diff == 0:
                min_max_diff = 1
            points[:, row] = (points[:, row] - min_row) / min_max_diff
        return points
    else:
        max_row = np.max(points[:])
        min_row = np.min(points[:])
        min_max_diff = max_row - min_row
        points[:] = (points[:] - min_row) / min_max_diff
        return points


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
        self.noise_sum = 1e-2
        self.original_points_x = self.points_x.copy()
        self.original_points_y = self.points_y.copy()
        self.is_normalized = False

    def stretch_points(self, start: ndarray, finish: ndarray, noise_level: float, count: int, seed: int):
        dist = finish - start
        np.random.seed(seed)
        a1 = np.random.rand(count, self.linear.n)
        temp = a1.copy() * dist + start
        self.points_x = np.array([np.concatenate((x, np.array([1]))) for x in temp])
        self.points_y = np.array(
            [self.linear(x) + noise_level * random.uniform(-1, 1)
             for x in self.points_x])
        self.noise_sum = 1e-2
        self.original_points_x = self.points_x.copy()
        self.original_points_y = self.points_y.copy()
        self.is_normalized = False
        str = ''
        for i in range(start.size):
            str += (f"({start[i]}, {finish[i]}), ")
        return str

    def draw(self):
        assert self.linear.n == 1
        x = self.points_x[:, 0]
        y = self.points_y
        plt.plot(x, y, '.')
        plt.show()

    def normalize_points(self):
        self.is_normalized = True
        self.points_x = normalized_points(self.points_x)

    def denormalize_points(self):
        self.points_x = self.original_points_x.copy()
        self.points_y = self.original_points_y.copy()
        self.is_normalized = False

    def denormalize_solution(self, solution: ndarray) -> ndarray:
        assert((not self.is_normalized), "graphic should be denormalized first with denormalize_points method")
        denormalized_solution = solution.copy()
        for i in range(self.points_x.shape[1] - 1):
            min_row = np.min(self.points_x[:, i])
            max_row = np.max(self.points_x[:, i])
            min_max_diff = max_row - min_row
            denormalized_solution[i] = solution[i] / min_max_diff
            denormalized_solution[-1] -= min_row * denormalized_solution[i]
        return denormalized_solution

class F(Graphic):
    def __init__(self):
        super().__init__(Linear(np.array([1, 2])), -5, 5, 0.13, 50, 121391674)


class G(Graphic):
    def __init__(self):
        super().__init__(Linear(np.array([2, 4])), -1, 1, 0.01, 50, 1)

    def __str__(self):
        return 'G'

class G_1(Graphic):
    def __init__(self):
        super().__init__(Linear(np.array([8, -1])), -1, 1, 0.01, 50, 1)

    def __str__(self):
        return 'G_1'


class F3d(Graphic):
    def __init__(self):
        super().__init__(Linear(np.array([1, 1, 2])), -1, 1, 0.01, 100, 28347834)


class F10d(Graphic):
    def __init__(self):
        super(F10d, self).__init__(Linear(
            np.array([1.1, 34, 2.1, 0.2343, 34.4, -1.3, 1, 12.2, 0.01, -0.01])), -1, 1, 0.01, 100, 364937374)
