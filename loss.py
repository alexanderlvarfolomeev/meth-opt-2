from abc import abstractmethod

import numpy as np
from numpy import ndarray, sign

from graphic import Linear


class Loss:
    @abstractmethod
    def loss_point(self, linear: Linear, x: ndarray, y: float) -> ndarray:
        pass

    def loss_points(self, linear: Linear, points_x: ndarray, points_y: ndarray) -> ndarray:
        loss = np.array([0] * linear.n)
        for i in range(points_x.shape[0]):
            loss = loss + self.loss_point(linear, points_x[i], points_y[i])
        return loss


class AbsoluteLoss(Loss):
    def loss_point(self, linear: Linear, x: ndarray, y: float) -> ndarray:
        return sign(linear(x) - y) * np.concatenate((x, np.array([1])))
