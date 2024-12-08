import torch


class CegisConfig:
    b1_act = ['SKIP']
    b1_hidden = [10]
    b2_act = ['SKIP']
    b2_hidden = [10]
    example = None
    bm1_hidden = []
    bm2_hidden = []
    rm1_hidden = []
    rm2_hidden = []
    bm1_act = []
    bm2_act = []
    rm1_act = []
    rm2_act = []
    batch_size = 500
    lr = 0.1
    loss_weight = (1, 1, 1, 1, 1, 1, 1, 1)
    loss_weight_continuous = (1, 1, 1)
    R_b = 0.5
    margin = 0.5
    DEG = [2] * 8
    DEG_continuous = [2] * 4
    split = False
    learning_loops = 100
    OPT = torch.optim.AdamW
    max_iter = 100
    bm1 = None
    bm2 = None
    rm1 = None
    rm2 = None
    counterexample_nums = 100
    lie_counterexample = 0
    counterexamples_ellipsoid = False
    eps = 0.05
    C_b = 0.2

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
