import timeit

import numpy as np
import torch
import sys
sys.path.append("/home/rmx/workspace/cav2025")
from Examplers import get_example_by_name
from learn.Cegis_barrier import Cegis
from utils.Config import CegisConfig


def main():
    start = timeit.default_timer()
    b1_activations = ['SKIP', 'SKIP']
    b1_hidden_neurons = [10] * len(b1_activations)

    example = get_example_by_name('C1')

    start = timeit.default_timer()
    opts = {
        'b1_act': b1_activations,
        'b1_hidden': b1_hidden_neurons,
        "example": example,
        'bm1_act': [],
        "batch_size": 100,
        'lr': 0.1,
        'loss_weight_continuous': (1, 1, 1),
        'R_b': 0.8,
        'margin': 2,
        "DEG_continuous": [2, 2, 1, 2],
        "learning_loops": 100,
        'max_iter': 10
    }
    Config = CegisConfig(**opts)
    cegis = Cegis(Config)
    end = cegis.solve()
    print('Elapsed Time: {}'.format(end - start))


if __name__ == '__main__':
    torch.manual_seed(2024)
    np.random.seed(2024)
    main()
