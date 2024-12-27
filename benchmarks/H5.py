import sys, os
sys.path.append("/home/rmx/workspace/cav2025_2/cav2025")

import timeit
import torch
import numpy as np
from utils.Config import CegisConfig
from Examplers import get_example_by_name, get_example_by_id
from learn.Learner import Learner
from learn.Cegis_barrier import Cegis


def main():
    start = timeit.default_timer()
    # B1
    b1_activations = ['SKIP']  # 'SKIP','SQUARE','MUL','LINEAR' are optional.
    b1_hidden_neurons = [20] * len(b1_activations)  # the number of hidden layer nodes.

    # B2
    b2_activations = ['SKIP']
    b2_hidden_neurons = [10] * len(b2_activations)

    example = get_example_by_name('H5')
    path = './output/H5/'
    if not os.path.isdir(path):
        os.mkdir(path)
        
    start = timeit.default_timer()
    opts = {
        'path':path,
        'b1_act': b1_activations,
        'b1_hidden': b1_hidden_neurons,
        'b2_act': b2_activations,
        'b2_hidden': b2_hidden_neurons,
        "example": example,
        # Multipliers for Lie derivative conditions.
        'bm1_hidden': [],  # the number of hidden layer nodes.
        'bm2_hidden': [],
        'bm1_act': [],  # the activation function.
        'bm2_act': [],
        # Multipliers for guard conditions.
        'rm1_hidden': [],  # the number of hidden layer nodes.
        'rm2_hidden': [],
        'rm1_act': [],  # the activation function.
        'rm2_act': [],
        # Neural network
        "batch_size": 1000,
        'lr': 0.2,  # the learning rate
        'loss_weight': (1, 1, 1, 1, 1, 1, 1, 1),  # The weight of the loss term
        'R_b': 0.6,
        'margin': 1,
        "learning_loops": 100,
        # Verification
        "DEG": [2, 2, 2, 2, 2, 2, 2, 2],  # Degrees of multipliers during SOS verification.
        'max_iter': 100,  # The maximum number of iterations.
        'counterexample_nums': 100  # The number of counterexamples generated each time.
    }
    Config = CegisConfig(**opts)
    cegis = Cegis(Config)
    end = cegis.solve()
    print('Elapsed Time: {}'.format(end - start))


if __name__ == '__main__':
    torch.manual_seed(2024)
    np.random.seed(2024)
    main()
