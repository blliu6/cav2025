import timeit
import torch
import numpy as np
import sys
import os
from loguru import logger
sys.path.append("/home/rmx/workspace/cav2025_2/cav2025")
from utils.Config import CegisConfig
from Examplers import get_example_by_name, get_example_by_id
from learn.Learner import Learner
from learn.Cegis_barrier import Cegis


def main(params=None):
    start = timeit.default_timer()
    b1_activations = ['SKIP', 'SKIP', 'SKIP']
    b1_hidden_neurons = [10] * len(b1_activations)

    example = get_example_by_name('NF1')

    path = './output/NF1/'
    if not os.path.isdir(path):
        os.mkdir(path)

    start = timeit.default_timer()
    opts = {
        'b1_act': b1_activations,
        'b1_hidden': b1_hidden_neurons,
        "example": example,
        'path': path,
        "batch_size": params['batch_size'],
        'lr': params['lr'],
        'loss_weight_continuous': params['loss_weight_continuous'],
        'R_b': params['R_b'],
        'margin': params['margin'],
        "DEG_continuous": params['DEG_continuous'],
        "learning_loops": params['learning_loops'],
        'max_iter': params['max_iter'],
        'err': params['err'],
    }
    Config = CegisConfig(**opts)
    cegis = Cegis(Config)
    end = cegis.solve()
    print('Elapsed Time: {}'.format(end - start))


if __name__ == '__main__':
    torch.manual_seed(2024)
    np.random.seed(2024)

    from datetime import datetime
    
    params = {
    "batch_size": 500,
        'lr': 0.1,
        'loss_weight_continuous': (1, 1, 1),
        'R_b': 0.4,
        'margin': 0.4,
        "DEG_continuous": [2, 4, 4, 2],
        "learning_loops": 100,
        'max_iter': 20,
        'err': 0.02677454477719443,
    }

    for batch_size in [500, 600, 700, 800, 900, 1000]:
        for lr in np.linspace(0.01, 0.3, 30):
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            sys.stdout = open(f"./log/NF1-{timestamp}.txt", "w")

            params['batch_size'] = batch_size
            params['lr'] = lr

            print(params)
            print()
            main(params)

            sys.stdout = sys.__stdout__

    