import numpy as np
from numpy import ndarray

from graphic import Graphic, Linear
from loss import Loss


def gradient(graphic: Graphic, w: ndarray, loss: Loss, epoches: int, batch_size: int) -> ndarray:
    empiric_risk = loss.loss_points(Linear(w), graphic.points_x, graphic.points_y) / graphic.points_x.shape[0]
    batches = (graphic.points_x.shape[0] + batch_size - 1) // batch_size
    for epoch in range(epoches):
        for batch_index in range(batches):
            indexes = [i for i in range(batch_index * batch_size, (batch_index + 1) * batch_size)]
            points_x = np.array([graphic.points_x[pos] for pos in indexes])
            points_y = np.array([graphic.points_y[pos] for pos in indexes])
            new_empiric_risk = loss.loss_points(Linear(w), points_x, points_y)
            w = w - 0.01 * new_empiric_risk
    return w
