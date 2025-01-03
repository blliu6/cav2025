import timeit
import os
import torch
import numpy as np
from utils.Config import CegisConfig
from Examplers import get_example_by_name, get_example_by_id
from learn.Learner import Learner
from learn.Cegis_barrier import Cegis


def main():
    start = timeit.default_timer()
    b1_activations = ['LINEAR']
    b1_hidden_neurons = [10] * len(b1_activations)

    example = get_example_by_name('C22')

    path = './output/C22/'
    if not os.path.isdir(path):
        os.mkdir(path)
    start = timeit.default_timer()
    opts = {
        'path': path,
        'b1_act': b1_activations,
        'b1_hidden': b1_hidden_neurons,
        "example": example,
        'bm1_act': [],
        "batch_size": 2000,
        'lr': 0.05,
        'loss_weight_continuous': (100, 1, 1),
        'R_b': 0.2,
        'margin': 1,
        'split': True,
        "DEG_continuous": [2, 2, 1, 2],
        "learning_loops": 100,
        'max_iter': 5
    }
    Config = CegisConfig(**opts)
    cegis = Cegis(Config)
    end = cegis.solve()
    print('Elapsed Time: {}'.format(end - start))


if __name__ == '__main__':
    torch.manual_seed(2024)
    np.random.seed(2024)
    main()
