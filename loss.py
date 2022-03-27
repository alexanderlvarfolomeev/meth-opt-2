from abc import abstractmethod

import numpy as np
from numpy import ndarray, sign

from graphic import Linear


class Loss:
    @abstractmethod
    def loss_point(self, linear: Linear, point: ndarray) -> ndarray:
        pass

    def loss_points(self, linear: Linear, points: ndarray) -> ndarray:
        loss = np.array([0, 0])
        for point in points:
            loss = loss + self.loss_point(linear, point)
        return loss


class AbsoluteLoss(Loss):
    def loss_point(self, linear: Linear, point: ndarray) -> ndarray:
        return sign(linear.w * point[0] + linear.b - point[1]) * np.array([point[0], 1])
