import numpy as np
import torch

from utils.Config import CegisConfig
from benchmarks.Examplers import Zone, Example


class Data:
    def __init__(self, config: CegisConfig):
        self.config = config
        self.ex = config.example
        self.n = self.ex.n

    def get_data(self, zone: Zone, batch_size):
        global s
        if zone.shape == 'box':
            times = 1 / (1 - self.config.R_b)
            s = np.clip((np.random.rand(batch_size, self.n) - 0.5) * times, -0.5, 0.5)
            center = (zone.low + zone.up) / 2
            s = s * (zone.up - zone.low) + center

        elif zone.shape == 'ball':
            s = np.random.randn(batch_size, self.n)
            s = np.array([e / np.sqrt(sum(e ** 2)) * np.sqrt(zone.r) for e in s])
            s = np.array(
                [e * np.random.random() ** (1 / self.n) if np.random.random() > self.config.C_b else e for e in s])
            s = s + zone.center

        # from matplotlib import pyplot as plt
        # plt.plot(s[:, :1], s[:, -1], '.')
        # plt.gca().set_aspect(1)
        # plt.show()
        return torch.Tensor(s)

    def x2dotx(self, X, f):
        f_x = []
        for x in X:
            f_x.append([f[i](x) for i in range(self.n)])
        return torch.Tensor(f_x)

    def generate_data(self):
        batch_size = self.config.batch_size
        l1 = self.get_data(self.ex.l1, batch_size)
        l2 = self.get_data(self.ex.l2, batch_size)
        I = self.get_data(self.ex.I, batch_size)
        U = self.get_data(self.ex.U, batch_size)
        g1 = self.get_data(self.ex.g1, batch_size)
        g2 = self.get_data(self.ex.g2, batch_size)

        l1_dot = self.x2dotx(l1, self.ex.f1)
        l2_dot = self.x2dotx(l2, self.ex.f2)
        return l1, l2, I, U, g1, g2, l1_dot, l2_dot

    def generate_data_for_continuous(self):
        batch_size = self.config.batch_size
        l1 = self.get_data(self.ex.l1, batch_size)
        I = self.get_data(self.ex.I, batch_size)
        U = self.get_data(self.ex.U, batch_size)

        l1_dot = self.x2dotx(l1, self.ex.f1)

        return l1, I, U, l1_dot
