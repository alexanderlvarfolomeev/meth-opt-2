import numpy as np
from matplotlib import pyplot as plt

from criteria import StopLossCriteria
from gradient import gradient
from graphic import F, F3d, G, F10d, Linear, Graphic
from loss import AbsoluteLoss


def main():
    print(gradient(F(), 0.01, np.array([0, 0]), AbsoluteLoss(), 1000, 1, StopLossCriteria(), 1e-4))

    # return
    f = Graphic(Linear(np.array([1, 2, 4])), -1, 1, 0.01, 50, 1)
    print(gradient(f, 0.01, (np.array([0] * (f.linear.n + 1))), AbsoluteLoss(), 51000, 1, StopLossCriteria(), 1e-4))

    return
    for graphic in [F(), G(), F3d(), F10d()]:
        count_points = graphic.points_x.shape[0]
        batch_sizes = [i for i in range(1, count_points + 1, 5)]
        w = np.array([0] * (graphic.linear.n + 1))
        epoches = np.array([gradient(graphic, 0.1, w, AbsoluteLoss(), 5000, batch_size, StopLossCriteria(), 1e-4)[0]
                            for batch_size in batch_sizes])
        plt.plot(batch_sizes, epoches)
        plt.show()


if __name__ == '__main__':
    main()
