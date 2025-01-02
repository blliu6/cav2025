import timeit
import os
import numpy as np
import torch
from loguru import logger
import sys

sys.path.append("/home/rmx/workspace/cav2025_2/cav2025")
from Examplers import get_example_by_name
from learn.Cegis_barrier import Cegis
from utils.Config import CegisConfig


def main():
    start = timeit.default_timer()
    b1_activations = ['SQUARE']
    b1_hidden_neurons = [10] * len(b1_activations)

    example = get_example_by_name('F0')

    start = timeit.default_timer()

    path = './output/F0/'
    if not os.path.isdir(path):
        os.mkdir(path)

    opts = {
        'b1_act': b1_activations,
        'b1_hidden': b1_hidden_neurons,
        "example": example,
        'bm1_act': [],
        "batch_size": 500,
        'lr': 0.5,
        'loss_weight_continuous': (1, 1, 1),
        'R_b': 0.5,
        'margin': 2,
        "DEG_continuous": [2, 2, 1, 2],
        "learning_loops": 100,
        'max_iter': 5,
        'path': path
    }
    Config = CegisConfig(**opts)
    cegis = Cegis(Config)
    end = cegis.solve()
    print('Elapsed Time: {}'.format(end - start))


if __name__ == '__main__':
    torch.manual_seed(2024)
    np.random.seed(2024)
    main()
