import numpy as np

from criteria import StopLossCriteria
from gradient import gradient
from graphic import F10d, F, G, F3d
from loss import AbsoluteLoss


def test():
    for graphic in [F(), G(), F3d(), F10d()]:
        print(graphic)
        count_points = graphic.points_x.shape[0]
        batch_sizes = [i for i in range(1, count_points + 1, 5)]
        w = np.array([0] * (graphic.linear.n + 1))
        for batch_size in batch_sizes:
            print(batch_size)
            t = gradient(graphic, 0.1, w, AbsoluteLoss(), 2000, batch_size, StopLossCriteria(), 1e-4)
            if t[0] < 2000 and (graphic.linear.a - t[1]).sum() > 0.1:
                print("Wrong answer")


if __name__ == '__main__':
    test()
