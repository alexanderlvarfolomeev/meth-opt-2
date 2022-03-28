from abc import abstractmethod

from numpy import ndarray

from graphic import Linear
from loss import Loss


class StopCriteria:
    @abstractmethod
    def stop(self, w: ndarray, loss: Loss, points_x: ndarray, points_y: ndarray, eps: float) -> bool:
        pass


class StopLossCriteria(StopCriteria):

    def stop(self, w: ndarray, loss: Loss, points_x: ndarray, points_y: ndarray, eps: float) -> bool:
        return loss.loss_points(Linear(w), points_x, points_y) < eps
