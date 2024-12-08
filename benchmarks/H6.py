import timeit
import torch
import numpy as np
from utils.Config import CegisConfig
from Examplers import get_example_by_name, get_example_by_id
from learn.Learner import Learner
from learn.Cegis_barrier import Cegis


def main():
    start = timeit.default_timer()
    b1_activations = ['SKIP']
    b1_hidden_neurons = [10] * len(b1_activations)

    b2_activations = ['SKIP']
    b2_hidden_neurons = [10] * len(b2_activations)

    example = get_example_by_name('H6')

    start = timeit.default_timer()
    opts = {
        'b1_act': b1_activations,
        'b1_hidden': b1_hidden_neurons,
        'b2_act': b2_activations,
        'b2_hidden': b2_hidden_neurons,
        "example": example,
        'bm1_hidden': [],
        'bm2_hidden': [],
        'bm1_act': [],
        'bm2_act': [],
        'rm1_hidden': [],
        'rm2_hidden': [],
        'rm1_act': [],
        'rm2_act': [],
        "batch_size": 500,
        'lr': 0.2,
        'loss_weight': (1, 1, 1, 1, 1, 1, 1, 1),
        'R_b': 0.5,
        'margin': 1,
        "DEG": [2, 2, 2, 2, 2, 2, 2, 2],
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
