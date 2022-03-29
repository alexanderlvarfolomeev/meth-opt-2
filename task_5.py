import numpy as np
from matplotlib import pyplot as plt
from numpy import ndarray

from criteria import CountStopLossCriteria
from gradient import gradient, NaiveGradient, Momentum, Nesterov, AdaGrad, RMSProp, Adam
from graphic import Graphic, Linear, F, G, G_1
from loss import AbsoluteLoss, Loss


class Norm_5d_Graphic(Graphic):
    def __init__(self):
        super().__init__(Linear(np.array([1, 4, 89, 34, 12])), -1, 1, 0.01, 30, seed=1)

    def get_all_stretches(self):
        return [
            # self.stretch_0,
            self.stretch_1,
        ]

    def stretch_0(self):
        start = np.array([0, 0, 0, 0])
        finish = np.array([1, 1, 1, 1])
        return self.stretch_points(start=start, finish=finish, noise_level=0.01, count=30, seed=1)

    def stretch_1(self):
        start = np.array([0, -20, 30, 0])
        finish = np.array([10, -10, 60, 5])
        return self.stretch_points(start=start, finish=finish, noise_level=0.01, count=30, seed=1)


class Norm_2d_Graphic(Graphic):
    def __init__(self):
        super().__init__(Linear(np.array([1, 4])), -1, 1, 0.01, 30, seed=1)

    def get_all_stretches(self):
        return [self.stretch_0, self.stretch_1, self.stretch_2, self.stretch_3]

    def stretch_0(self):
        start = np.array([0])
        finish = np.array([1])
        return self.stretch_points(start=start, finish=finish, noise_level=0.01, count=30, seed=1)

    def stretch_1(self):
        start = np.array([-1])
        finish = np.array([1])
        return self.stretch_points(start=start, finish=finish, noise_level=0.01, count=30, seed=1)

    def stretch_2(self):
        start = np.array([-5])
        finish = np.array([5])
        return self.stretch_points(start=start, finish=finish, noise_level=0.01, count=30, seed=1)

    def stretch_3(self):
        start = np.array([1])
        finish = np.array([40])
        return self.stretch_points(start=start, finish=finish, noise_level=0.01, count=30, seed=1)


class Norm_3d_Graphic(Graphic):
    def __init__(self):
        super().__init__(Linear(np.array([1, 4, 13])), -1, 1, 0.01, 30, seed=1)

    def get_all_stretches(self):
        return [self.stretch_0, self.stretch_1, self.stretch_2]

    def stretch_0(self):
        start = np.array([-1, -1])
        finish = np.array([1, 1])
        return self.stretch_points(start=start, finish=finish, noise_level=0.01, count=30, seed=1)

    def stretch_1(self):
        start = np.array([-10, 30])
        finish = np.array([10, 50])
        return self.stretch_points(start=start, finish=finish, noise_level=0.01, count=30, seed=1)

    def stretch_2(self):
        start = np.array([-5, 100])
        finish = np.array([5, 130])
        return self.stretch_points(start=start, finish=finish, noise_level=0.01, count=30, seed=1)


class Norm_10d_Graphic(Graphic):
    def __init__(self):
        super().__init__(Linear(np.array([1, 4, 13, 334, 32, 11, 0, 7, 63, 10])), -1, 1, 0.01, 30, seed=1)

    def get_all_stretches(self):
        return [self.stretch_0, self.stretch_1, self.stretch_2]

    def stretch_0(self):
        start = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        finish = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        return self.stretch_points(start=start, finish=finish, noise_level=0.01, count=30, seed=1)

    def stretch_1(self):
        start = np.array([-1, -1, -1, -1, -1, -1, -1, -1, -1, -1])
        finish = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        return self.stretch_points(start=start, finish=finish, noise_level=0.01, count=30, seed=1)

    def stretch_2(self):
        start = np.array([10, 2, -40, 3, -80, 1, 0, -90, 100, 23])
        finish = np.array([20, 5, -20, 43, 80, 11, 18, -70, 104, 25])
        return self.stretch_points(start=start, finish=finish, noise_level=0.01, count=30, seed=1)


class Solver:
    def __init__(self, loss: Loss, points_x: ndarray, points_y: ndarray):
        self.loss = loss
        self.points_x = points_x
        self.points_y = points_y

    def __call__(self, X: ndarray, Y: ndarray):
        Z = X.copy()
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                x = X[i][j]
                y = Y[i][j]
                Z[i][j] = self.one_call(x, y)
        return Z

    def one_call(self, x, y):
        return self.loss.loss_points(Linear(np.array([x, y])), self.points_x, self.points_y)


def draw_steps(points: ndarray, loss: Loss, points_x: ndarray, points_y: ndarray, subtitle):
    x = points[:, 0]
    y = points[:, 1]
    const = 1
    x_start = x.min() - const
    x_end = x.max() + const
    y_start = y.min() - const
    y_end = y.max() + const
    x_min = x.min()
    x_max = x.max()
    y_min = y.min()
    y_max = y.max()
    x_lin = np.linspace(x_start, x_end, 100)
    y_lin = np.linspace(y_start, y_end, 100)
    X, Y = np.meshgrid(x_lin, y_lin)
    solver: Solver = Solver(loss, points_x, points_y)
    Z = solver(X, Y)
    fig = plt.figure()
    plt.plot(points[:, 0], points[:, 1], ".-")
    plt.contour(X, Y, Z, levels=100)
    fig.suptitle(subtitle)
    plt.show()
    plt.close(fig)


def draw_analisys():
    loss: Loss = AbsoluteLoss()
    batch_sizes = [1, 3, 5, 10, 20, 30]
    for graphic in [G_1(), G()]:
        for batch_size in batch_sizes:
            n = graphic.linear.n + 1
            for grad in [NaiveGradient(), Momentum(n, 0.9), Nesterov(n, 0.9),
                         AdaGrad(n), RMSProp(n, 0.99), Adam(n, 0.9, 0.99)]:
                w = np.array([0] * (graphic.linear.n + 1))
                info = f"{str(graphic)} function, batch_size={batch_size}, {str(grad)}"
                arrr = []
                max_iters = 6000
                eps_value = 0.001
                gradient(graphic, 0.1, w, AbsoluteLoss(), max_iters,
                         batch_size, CountStopLossCriteria(5), eps=eps_value
                         , grad=grad,
                         base_step=5,
                         use_base_step=False,
                         array_of_steps_points=arrr)
                points_way = np.array(arrr)
                draw_steps(points_way, loss, graphic.points_x, graphic.points_y, info)


if __name__ == '__main__':
    draw_analisys()
