import torch
import torch.nn as nn
import numpy as np
import sympy as sp
from utils.Config import CegisConfig
from torch.func import jacrev


class Net(nn.Module):
    def __init__(self, config: CegisConfig):
        super(Net, self).__init__()
        self.config = config
        self.input = config.example.n
        self.b1_lay1, self.b1_lay2 = [], []
        self.bm1_lay1, self.bm1_lay2 = [], []

        #############################################################
        # b1网络
        n_prev = self.input
        k = 1
        for n_hid, act in zip(config.b1_hidden, config.b1_act):
            layer1 = nn.Linear(n_prev, n_hid)

            if act == 'SKIP':
                layer2 = nn.Linear(self.input, n_hid)
            else:
                layer2 = nn.Linear(n_prev, n_hid)

            self.register_parameter(f'b1_w1_{k}', layer1.weight)
            self.register_parameter(f'b1_w2_{k}', layer2.weight)

            self.register_parameter(f'b1_b1_{k}', layer1.bias)
            self.register_parameter(f'b1_b2_{k}', layer2.bias)

            self.b1_lay1.append(layer1)
            self.b1_lay2.append(layer2)
            n_prev = n_hid
            k = k + 1

        layer1 = nn.Linear(n_prev, 1, bias=False)
        self.register_parameter(f'b1_w1_{k}', layer1.weight)
        self.b1_lay1.append(layer1)
        #############################################################

        #############################################################
        # bm1_乘子网络
        if len(config.bm1_hidden) == 0:
            if config.bm1 is not None:
                bm1 = nn.Parameter(torch.Tensor([config.bm1]), requires_grad=False)
            else:
                bm1 = nn.Parameter(torch.randn(1))
            self.register_parameter('bm1', bm1)
            self.bm1_lay1.append(bm1)
        else:
            n_prev = self.input
            k = 1
            for n_hid, act in zip(config.bm1_hidden, config.bm1_act):
                layer1 = nn.Linear(n_prev, n_hid)

                if act == 'SKIP':
                    layer2 = nn.Linear(self.input, n_hid)
                else:
                    layer2 = nn.Linear(n_prev, n_hid)

                self.register_parameter(f'bm1_w1_{k}', layer1.weight)
                self.register_parameter(f'bm1_w2_{k}', layer2.weight)

                self.register_parameter(f'bm1_b1_{k}', layer1.bias)
                self.register_parameter(f'bm1_b2_{k}', layer2.bias)

                self.bm1_lay1.append(layer1)
                self.bm1_lay2.append(layer2)
                n_prev = n_hid
                k = k + 1

            layer1 = nn.Linear(n_prev, 1)
            self.register_parameter(f'bm1_w1_{k}', layer1.weight)
            self.register_parameter(f'bm1_b1_{k}', layer1.bias)
            self.bm1_lay1.append(layer1)
        #############################################################

    def forward(self, x):
        y, act = x, self.config.b1_act
        for idx, (layer1, layer2) in enumerate(zip(self.b1_lay1[:-1], self.b1_lay2)):
            if act[idx] == 'SQUARE':
                y = (layer1(y)) ** 2
            elif act[idx] == 'SKIP':
                z1 = layer1(y)
                z2 = layer2(x)
                y = z1 * z2
            elif act[idx] == 'MUL':
                z1 = layer1(y)
                z2 = layer2(y)
                y = z1 * z2
            elif act[idx] == 'LINEAR':
                y = layer1(y)
        y = self.b1_lay1[-1](y)
        return y
        # l1, I, U, l1_dot = data
        # #############################################################
        # # loss 1
        # b1_y = self.net_out(I, self.config.b1_act, self.b1_lay1, self.b1_lay2)
        # #############################################################
        # # loss 2
        # bl_1 = self.net_out(l1, self.config.b1_act, self.b1_lay1, self.b1_lay2)
        # b1_grad = self.get_gradient(l1, l1_dot, self.config.b1_act, self.b1_lay1, self.b1_lay2)
        # # bl_1, b1_grad = self.get_out_and_grad(l1, l1_dot, self.config.b1_act, self.b1_lay1, self.b1_lay2)
        #
        # if len(self.config.bm1_hidden) == 0:
        #     bm1_y = l1 * 0 + self.bm1_lay1[0]
        # else:
        #     bm1_y = self.net_out(l1, self.config.bm1_act, self.bm1_lay1, self.bm1_lay2)
        # #############################################################
        # # loss 8
        # b2_y = self.net_out(U, self.config.b1_act, self.b1_lay1, self.b1_lay2)
        #
        # return b1_y, bl_1, b1_grad, bm1_y, b2_y

    def net_out(self, x, act, lay1, lay2):
        y = x
        for idx, (layer1, layer2) in enumerate(zip(lay1[:-1], lay2)):
            if act[idx] == 'SQUARE':
                y = (layer1(y)) ** 2
            elif act[idx] == 'SKIP':
                z1 = layer1(y)
                z2 = layer2(x)
                y = z1 * z2
            elif act[idx] == 'MUL':
                z1 = layer1(y)
                z2 = layer2(y)
                y = z1 * z2
            elif act[idx] == 'LINEAR':
                y = layer1(y)
        y = lay1[-1](y)
        return y

    def get_out_and_grad(self, x, xdot):
        act, lay1, lay2 = self.config.b1_act, self.b1_lay1, self.b1_lay2
        y = x
        jacobian = torch.diag_embed(torch.ones(x.shape[0], self.input))
        for idx, (layer1, layer2) in enumerate(zip(lay1[:-1], lay2)):
            if act[idx] == 'SQUARE':
                z = layer1(y)
                y = z ** 2
                jacobian = torch.matmul(torch.matmul(2 * torch.diag_embed(z), layer1.weight), jacobian)
            elif act[idx] == 'SKIP':
                z1 = layer1(y)
                z2 = layer2(x)
                y = z1 * z2
                jacobian = torch.matmul(torch.diag_embed(z1), layer2.weight) + torch.matmul(
                    torch.matmul(torch.diag_embed(z2), layer1.weight), jacobian)
            elif act[idx] == 'MUL':
                z1 = layer1(y)
                z2 = layer2(y)
                y = z1 * z2
                grad = torch.matmul(torch.diag_embed(z1), layer2.weight) + torch.matmul(torch.diag_embed(z2),
                                                                                        layer1.weight)
                jacobian = torch.matmul(grad, jacobian)
            elif act[idx] == 'LINEAR':
                z = layer1(y)
                y = z
                jacobian = torch.matmul(layer1.weight, jacobian)

        y = lay1[-1](y)
        jacobian = torch.matmul(lay1[-1].weight, jacobian)
        grad_y = torch.sum(torch.mul(jacobian[:, 0, :], xdot), dim=1)
        return y, grad_y

    # def transform_data(self, data, f):
    #     ans = [torch.unsqueeze(torch.tensor(list(map(ff, data))), dim=1) for ff in f]
    #     return torch.cat(ans, dim=1)
    def get_bm1(self, l1):
        if len(self.config.bm1_hidden) == 0:
            bm1_y = torch.zeros((len(l1), 1)) + self.bm1_lay1[0]
        else:
            bm1_y = self.net_out(l1, self.config.bm1_act, self.bm1_lay1, self.bm1_lay2)
        return bm1_y

    def get_gradient(self, x, xdot):
        act, lay1, lay2 = self.config.b1_act, self.b1_lay1, self.b1_lay2
        y = x
        jacobian = torch.diag_embed(torch.ones(x.shape[0], self.input))
        for idx, (layer1, layer2) in enumerate(zip(lay1[:-1], lay2)):
            if act[idx] == 'SQUARE':
                z = layer1(y)
                y = z ** 2
                jacobian = torch.matmul(torch.matmul(2 * torch.diag_embed(z), layer1.weight), jacobian)
            elif act[idx] == 'SKIP':
                z1 = layer1(y)
                z2 = layer2(x)
                y = z1 * z2
                jacobian = torch.matmul(torch.diag_embed(z1), layer2.weight) + torch.matmul(
                    torch.matmul(torch.diag_embed(z2), layer1.weight), jacobian)
            elif act[idx] == 'MUL':
                z1 = layer1(y)
                z2 = layer2(y)
                y = z1 * z2
                grad = torch.matmul(torch.diag_embed(z1), layer2.weight) + torch.matmul(torch.diag_embed(z2),
                                                                                        layer1.weight)
                jacobian = torch.matmul(grad, jacobian)
            elif act[idx] == 'LINEAR':
                z = layer1(y)
                y = z
                jacobian = torch.matmul(layer1.weight, jacobian)

        jacobian = torch.matmul(lay1[-1].weight, jacobian)
        grad_y = torch.sum(torch.mul(jacobian[:, 0, :], xdot), dim=1)
        return grad_y

    def get_barriers(self):
        expr = []
        x = sp.symbols([['x{}'.format(i + 1) for i in range(self.input)]])
        #############################################################
        # 第一个
        expr_1 = self.sp_net(x, self.config.b1_act, self.b1_lay1, self.b1_lay2)
        expr.append(expr_1)
        #############################################################
        # 第三个 bm1
        if len(self.config.bm1_hidden) == 0:
            expr_3 = self.bm1_lay1[0].detach().numpy()[0]
        else:
            expr_3 = self.sp_net(x, self.config.bm1_act, self.bm1_lay1, self.bm1_lay2)
        expr.append(expr_3)

        return expr

    def sp_net(self, x, act, lay1, lay2):
        y = x
        for idx, (layer1, layer2) in enumerate(zip(lay1[:-1], lay2)):
            if act[idx] == 'SQUARE':
                w1 = layer1.weight.detach().numpy()
                b1 = layer1.bias.detach().numpy()
                z = np.dot(y, w1.T) + b1
                y = z ** 2
            elif act[idx] == 'SKIP':
                w1 = layer1.weight.detach().numpy()
                b1 = layer1.bias.detach().numpy()
                z1 = np.dot(y, w1.T) + b1

                w2 = layer2.weight.detach().numpy()
                b2 = layer2.bias.detach().numpy()
                z2 = np.dot(x, w2.T) + b2
                y = np.multiply(z1, z2)
            elif act[idx] == 'MUL':
                w1 = layer1.weight.detach().numpy()
                b1 = layer1.bias.detach().numpy()
                z1 = np.dot(y, w1.T) + b1

                w2 = layer2.weight.detach().numpy()
                b2 = layer2.bias.detach().numpy()
                z2 = np.dot(y, w2.T) + b2

                y = np.multiply(z1, z2)
            elif act[idx] == 'LINEAR':
                w1 = layer1.weight.detach().numpy()
                b1 = layer1.bias.detach().numpy()
                y = np.dot(y, w1.T) + b1

        if lay1[-1].__getattr__('bias') is None:
            w1 = lay1[-1].weight.detach().numpy()
            y = np.dot(y, w1.T)
        else:
            w1 = lay1[-1].weight.detach().numpy()
            b1 = lay1[-1].bias.detach().numpy()
            y = np.dot(y, w1.T) + b1
        y = sp.expand(y[0, 0])
        return y
