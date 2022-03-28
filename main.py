import numpy as np
from matplotlib import pyplot as plt

from criteria import StopLossCriteria
from gradient import gradient
from graphic import Linear, F
from loss import AbsoluteLoss


def main():
    F().draw()
    w = gradient(F(), 0.1, np.array([0, 0]), AbsoluteLoss(), 1000, 1, StopLossCriteria(), 1e-5)
    plt.plot([-1, Linear(w[1])(np.array([-1]))], [1, Linear(w[1])(np.array([1]))])
    print(w)


if __name__ == '__main__':
    main()
