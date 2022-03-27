import random

import numpy as np
from numpy import ndarray

from graphic import Graphic, Linear
from loss import Loss


def gradient(graphic: Graphic, w: ndarray, loss: Loss, epoches: int, batch_size: int, randomic: bool) -> ndarray:
    empiric_risk = loss.loss_points(Linear(w[0], w[1]), graphic.points) / graphic.points.shape[0]
    batches = (graphic.points.shape[0] + batch_size - 1) // batch_size
    for epoch in range(epoches):
        if randomic:
            for i in range(batches):
                indexes = [random.randint(0, graphic.points.shape[0] - 1) for _ in range(batch_size)]
                eta = 7 / (i + 0.01) + 0.1
                points = np.array([graphic.points[pos] for pos in indexes])
                new_empiric_risk = loss.loss_points(Linear(w[0], w[1]), points)
                w = w - eta * new_empiric_risk
        else:
            for batch_index in range(batches):
                indexes = [i for i in range(batch_index * batch_size, (batch_index + 1) * batch_size)]
                points = np.array([graphic.points[pos] for pos in indexes])
                new_empiric_risk = loss.loss_points(Linear(w[0], w[1]), points)
                w = w - 0.01 * new_empiric_risk
    return w
