import numpy as np
from matplotlib import pyplot as plt

from gradient import gradient
from graphic import Linear, G
from loss import AbsoluteLoss


def main():
    G().draw()
    w = gradient(G(), np.array([0, 0]), AbsoluteLoss(), 1000, 1, False)
    plt.plot(Linear(w[0], w[1])(-1), Linear(w[0], w[1])(1))
    print(w)


if __name__ == '__main__':
    main()
