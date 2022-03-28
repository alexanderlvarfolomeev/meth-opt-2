import numpy as np
from matplotlib import pyplot as plt

from gradient import gradient
from graphic import Linear, F
from loss import AbsoluteLoss


def main():
    F().draw()
    w = gradient(F(), np.array([0, 0]), AbsoluteLoss(), 10000, 1)
    plt.plot(Linear(w)(np.array([-1])), Linear(w)(np.array([1])))
    print(w)


if __name__ == '__main__':
    main()
