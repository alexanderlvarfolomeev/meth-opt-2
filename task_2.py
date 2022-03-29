import numpy as np
from matplotlib import pyplot as plt

from criteria import CountStopLossCriteria
from gradient import gradient
from graphic import Graphic, Linear
from loss import AbsoluteLoss


class Norm_2d_Graphic(Graphic):
    def __init__(self):
        super().__init__(Linear(np.array([1, 4])), -1, 1, 0.01, 30, seed=1)

    def __str__(self):
        return 'Norm_2d_Graphic'

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

    def __str__(self):
        return 'Norm_3d_Graphic'

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


class Norm_4d_Graphic(Graphic):
    def __init__(self):
        super().__init__(Linear(np.array([1, 4, 89, 34])), -1, 1, 0.01, 30, seed=1)

    def __str__(self):
        return 'Norm_4d_Graphic'

    def get_all_stretches(self):
        return [
            self.stretch_0,
            self.stretch_1,
            self.stretch_2,
        ]

    def stretch_0(self):
        start = np.array([0, 0, 0])
        finish = np.array([1, 1, 1])
        return self.stretch_points(start=start, finish=finish, noise_level=0.01, count=30, seed=1)

    def stretch_1(self):
        start = np.array([0, 1, 2])
        finish = np.array([1, 2, 3])
        return self.stretch_points(start=start, finish=finish, noise_level=0.01, count=30, seed=1)

    def stretch_2(self):
        start = np.array([0, 10, -1])
        finish = np.array([10, 20, 0])
        return self.stretch_points(start=start, finish=finish, noise_level=0.01, count=10, seed=1)


class Norm_10d_Graphic(Graphic):
    def __init__(self):
        super().__init__(Linear(np.array([1, 4, 13, 334, 32, 11, 0, 7, 63, 10])), -1, 1, 0.01, 30, seed=1)

    def __str__(self):
        return 'Norm_10d_Graphic'

    def get_all_stretches(self):
        return [self.stretch_0, self.stretch_1, self.stretch_2]

    def stretch_0(self):
        start = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])
        finish = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1])
        return self.stretch_points(start=start, finish=finish, noise_level=0.01, count=30, seed=1)

    def stretch_1(self):
        start = np.array([-1, -1, -1, -1, -1, -1, -1, -1, -1])
        finish = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1])
        return self.stretch_points(start=start, finish=finish, noise_level=0.01, count=30, seed=1)

    def stretch_2(self):
        start = np.array([10, 2, -40, 3, -80, 1, 0, -90, 100])
        finish = np.array([20, 5, -20, 43, 80, 11, 18, -70, 104])
        return self.stretch_points(start=start, finish=finish, noise_level=0.01, count=30, seed=1)


def analise(print_log: bool = False):
    np.set_printoptions(suppress=True)
    for graphic in [
        Norm_2d_Graphic(),
        Norm_3d_Graphic(),
        Norm_4d_Graphic(),
        Norm_10d_Graphic(),
    ]:
        for i, stretch in enumerate(graphic.get_all_stretches()):
            y_steps_no_normalization = np.empty(0)
            y_steps_normalization = np.empty(0)
            count_points = graphic.points_x.shape[0]
            batch_sizes = [i for i in range(1, count_points + 1, 2)]
            eps_value = 0.1
            cur_stretch = stretch()
            max_iters = 8000
            for batch_size in batch_sizes:
                w = np.array([0] * (graphic.linear.n + 1))
                non_norm_step, non_norm_res = gradient(graphic, 0.1, w, AbsoluteLoss(), max_iters,
                                                       batch_size, CountStopLossCriteria(5), eps=eps_value,
                                                       use_base_step=False)
                y_steps_no_normalization = np.append(y_steps_no_normalization, non_norm_step)
                graphic.normalize_points()
                normalized_steps, w_res = gradient(graphic, 0.1, w, AbsoluteLoss(), max_iters,
                                                   batch_size, CountStopLossCriteria(5), eps=eps_value
                                                   , base_step=5, use_base_step=False)
                graphic.denormalize_points()
                y_steps_normalization = np.append(y_steps_normalization, normalized_steps)
                w_denormalized = graphic.denormalize_solution(w_res)
                if print_log:
                    print(batch_size)
                    print((non_norm_step, non_norm_res))
                    print((normalized_steps, w_denormalized))
            fig = plt.figure()
            plt.plot(batch_sizes, y_steps_no_normalization, color='red', label='without normalization')
            plt.plot(batch_sizes, y_steps_normalization, color='green', label='with normalization')
            plt.xlabel('batch size')
            plt.ylabel('steps number')
            plt.legend(loc='best')
            fig.suptitle(f"Stretch: {cur_stretch}")
            plt.show()
            plt.close(fig)


if __name__ == '__main__':
    analise(print_log=True)
