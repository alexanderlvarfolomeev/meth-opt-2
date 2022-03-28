import random
from typing import Tuple, Any

import numpy as np
from numpy import ndarray

from criteria import StopCriteria
from graphic import Graphic, Linear
from loss import Loss


def gradient(graphic: Graphic,
             learning_rate: float,
             w: ndarray,
             loss: Loss,
             epoches: int,
             batch_size: int,
             criteria: StopCriteria,
             eps: float
             ) -> Tuple[int, Any]:
    batch_number = (graphic.points_x.shape[0] + batch_size - 1) // batch_size
    for epoch in range(epoches):
        temp = [[graphic.points_x[i], graphic.points_y[i]] for i in range(graphic.points_x.shape[0])]
        random.shuffle(temp)
        graphic.points_x = np.array([p[0] for p in temp])
        graphic.points_y = np.array([p[1] for p in temp])
        for batch_index in range(batch_number):
            indexes = [i for i in
                       range(batch_index * batch_size, min(graphic.points_x.shape[0], (batch_index + 1) * batch_size))]
            points_x = np.array([graphic.points_x[pos] for pos in indexes])
            points_y = np.array([graphic.points_y[pos] for pos in indexes])
            new_empiric_risk = loss.loss_gradient_points(Linear(w), points_x, points_y)
            w = w - learning_rate * new_empiric_risk
            if criteria.stop(w, loss, points_x, points_y, eps):
                return epoch, w
    return epoches, w
