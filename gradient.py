import random
from abc import abstractmethod
from typing import Tuple, Any

import numpy as np
from numpy import ndarray

from criteria import StopCriteria
from graphic import Graphic, Linear
from loss import Loss


class Gradient:
    @abstractmethod
    def compute_weights_diff(self, points_x: ndarray, points_y: ndarray, w: ndarray, loss: Loss) -> ndarray:
        pass

    @abstractmethod
    def get_memory_cost(self, batch_size: int) -> int:
        pass


class NaiveGradient(Gradient):
    def compute_weights_diff(self, points_x: ndarray, points_y: ndarray, w: ndarray, loss: Loss) -> ndarray:
        return loss.loss_gradient_points(Linear(w), points_x, points_y)

    def get_memory_cost(self, batch_size: int) -> int:
        return 0

    def __str__(self):
        return "NaiveGradient()"


class Momentum(Gradient):
    def __init__(self, n: int, b: float):
        self.b = b
        self.previous_gradient = np.zeros(n)

    def compute_weights_diff(self, points_x: ndarray, points_y: ndarray, w: ndarray, loss: Loss) -> ndarray:
        prev = self.previous_gradient
        self.previous_gradient = loss.loss_gradient_points(Linear(w), points_x, points_y)
        return self.b * prev + (1 - self.b) * self.previous_gradient

    def get_memory_cost(self, batch_size: int) -> int:
        return self.previous_gradient.size * 2

    def __str__(self):
        return f"Momentum({self.b})"


class Nesterov(Gradient):
    def __init__(self, n: int, b: float):
        self.b = b
        self.previous_gradient = np.zeros(n)

    def compute_weights_diff(self, points_x: ndarray, points_y: ndarray, w: ndarray, loss: Loss) -> ndarray:
        shift = self.b * self.previous_gradient
        self.previous_gradient = loss.loss_gradient_points(Linear(w - 0.1 * shift), points_x, points_y)
        return shift + (1 - self.b) * self.previous_gradient

    def get_memory_cost(self, batch_size: int) -> int:
        return self.previous_gradient.size * 3

    def __str__(self):
        return f"Nesterov({self.b})"


class AdaGrad(Gradient):
    def __init__(self, n: int):
        self.gradient_accumulator = np.zeros(n)

    def compute_weights_diff(self, points_x: ndarray, points_y: ndarray, w: ndarray, loss: Loss) -> ndarray:
        grad = loss.loss_gradient_points(Linear(w), points_x, points_y)
        self.gradient_accumulator += grad ** 2
        return grad / np.sqrt(self.gradient_accumulator + 1e-8)

    def get_memory_cost(self, batch_size: int) -> int:
        return self.gradient_accumulator.size * 2

    def __str__(self):
        return "AdaGrad()"


class RMSProp(Gradient):
    def __init__(self, n: int, g: float):
        self.g = g
        self.gradient_accumulator = np.zeros(n)

    def compute_weights_diff(self, points_x: ndarray, points_y: ndarray, w: ndarray, loss: Loss) -> ndarray:
        grad = loss.loss_gradient_points(Linear(w), points_x, points_y)
        self.gradient_accumulator = self.g * self.gradient_accumulator + (1 - self.g) * grad ** 2
        return grad / np.sqrt(self.gradient_accumulator + 1e-8)

    def get_memory_cost(self, batch_size: int) -> int:
        return self.gradient_accumulator.size * 3

    def __str__(self):
        return f"RMSProp({self.g})"


class Adam(Gradient):
    def __init__(self, n: int, b: float, g: float):
        self.b = b
        self.previous_gradient = np.zeros(n)
        self.g = g
        self.gradient_accumulator = np.zeros(n)

    def compute_weights_diff(self, points_x: ndarray, points_y: ndarray, w: ndarray, loss: Loss) -> ndarray:
        grad = loss.loss_gradient_points(Linear(w), points_x, points_y)
        self.previous_gradient = self.b * self.previous_gradient + (1 - self.b) * grad
        self.gradient_accumulator = self.g * self.gradient_accumulator + (1 - self.g) * grad ** 2
        return self.previous_gradient / np.sqrt(self.gradient_accumulator + 1e-8)

    def get_memory_cost(self, batch_size: int) -> int:
        return self.gradient_accumulator.size * 5

    def __str__(self):
        return f"Adam({self.b}, {self.g})"


def gradient(graphic: Graphic,
             learning_rate: float,
             w: ndarray,
             loss: Loss,
             epoches: int,
             batch_size: int,
             criteria: StopCriteria,
             eps: float,
             grad: Gradient = NaiveGradient(),
             array_of_steps_points=None,
             base_step: float = 100,
             use_base_step: bool = True
             ) -> Tuple[int, Any]:
    batch_number = (graphic.points_x.shape[0] + batch_size - 1) // batch_size
    for epoch in range(epoches):
        if not array_of_steps_points is None:
            array_of_steps_points.append(w)
        temp = [[graphic.points_x[i], graphic.points_y[i]] for i in range(graphic.points_x.shape[0])]
        random.shuffle(temp)
        graphic.points_x = np.array([p[0] for p in temp])
        graphic.points_y = np.array([p[1] for p in temp])
        for batch_index in range(batch_number):
            indexes = [i for i in
                       range(batch_index * batch_size, min(graphic.points_x.shape[0], (batch_index + 1) * batch_size))]
            points_x = np.array([graphic.points_x[pos] for pos in indexes])
            points_y = np.array([graphic.points_y[pos] for pos in indexes])
            if use_base_step:
                w = w - learning_rate * base_step * grad.compute_weights_diff(points_x, points_y, w, loss) / (epoch + 1)
            else:
                w = w - learning_rate * grad.compute_weights_diff(points_x, points_y, w, loss)  # origin version
            if criteria.stop(w, loss, graphic.points_x, graphic.points_y, eps):
                return epoch, w
    return epoches, w
