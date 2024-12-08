import torch
import torch.nn as nn
import numpy as np
import sympy as sp
from utils.Config import CegisConfig


class Net(nn.Module):
    def __init__(self, config: CegisConfig):
        super(Net, self).__init__()
        self.config = config
        self.input = config.example.n
        self.b1_lay1, self.b1_lay2 = [], []
        self.b2_lay1, self.b2_lay2 = [], []
        self.bm1_lay1, self.bm1_lay2 = [], []
        self.bm2_lay1, self.bm2_lay2 = [], []
        self.rm1_lay1, self.rm1_lay2 = [], []
        self.rm2_lay1, self.rm2_lay2 = [], []

        #############################################################
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
        n_prev = self.input
        k = 1
        for n_hid, act in zip(config.b2_hidden, config.b2_act):
            layer1 = nn.Linear(n_prev, n_hid)

            if act == 'SKIP':
                layer2 = nn.Linear(self.input, n_hid)
            else:
                layer2 = nn.Linear(n_prev, n_hid)

            self.register_parameter(f'b2_w1_{k}', layer1.weight)
            self.register_parameter(f'b2_w2_{k}', layer2.weight)

            self.register_parameter(f'b2_b1_{k}', layer1.bias)
            self.register_parameter(f'b2_b2_{k}', layer2.bias)

            self.b2_lay1.append(layer1)
            self.b2_lay2.append(layer2)
            n_prev = n_hid
            k = k + 1

        layer1 = nn.Linear(n_prev, 1, bias=False)
        self.register_parameter(f'b2_w1_{k}', layer1.weight)
        self.b2_lay1.append(layer1)
        #############################################################

        #############################################################
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

        #############################################################
        if len(config.bm2_hidden) == 0:
            if config.bm2 is not None:
                bm2 = nn.Parameter(torch.Tensor([config.bm2]), requires_grad=False)
            else:
                bm2 = nn.Parameter(torch.randn(1))
            self.register_parameter('bm2', bm2)
            self.bm2_lay1.append(bm2)
        else:
            n_prev = self.input
            k = 1
            for n_hid, act in zip(config.bm2_hidden, config.bm2_act):
                layer1 = nn.Linear(n_prev, n_hid)

                if act == 'SKIP':
                    layer2 = nn.Linear(self.input, n_hid)
                else:
                    layer2 = nn.Linear(n_prev, n_hid)

                self.register_parameter(f'bm2_w1_{k}', layer1.weight)
                self.register_parameter(f'bm2_w2_{k}', layer2.weight)

                self.register_parameter(f'bm2_b1_{k}', layer1.bias)
                self.register_parameter(f'bm2_b2_{k}', layer2.bias)

                self.bm2_lay1.append(layer1)
                self.bm2_lay2.append(layer2)
                n_prev = n_hid
                k = k + 1

            layer1 = nn.Linear(n_prev, 1)
            self.register_parameter(f'bm2_w1_{k}', layer1.weight)
            self.register_parameter(f'bm2_b1_{k}', layer1.bias)
            self.bm2_lay1.append(layer1)
        #############################################################

        #############################################################
        if len(config.rm1_hidden) == 0:
            if config.rm1 is not None:
                rm1 = nn.Parameter(torch.Tensor([config.rm1]), requires_grad=False)
            else:
                rm1 = nn.Parameter(torch.randn(1))
            self.register_parameter('rm1', rm1)
            self.rm1_lay1.append(rm1)
        else:
            n_prev = self.input
            k = 1
            for n_hid, act in zip(config.rm1_hidden, config.rm1_act):
                layer1 = nn.Linear(n_prev, n_hid)

                if act == 'SKIP':
                    layer2 = nn.Linear(self.input, n_hid)
                else:
                    layer2 = nn.Linear(n_prev, n_hid)

                self.register_parameter(f'rm1_w1_{k}', layer1.weight)
                self.register_parameter(f'rm1_w2_{k}', layer2.weight)

                self.register_parameter(f'rm1_b1_{k}', layer1.bias)
                self.register_parameter(f'rm1_b2_{k}', layer2.bias)

                self.rm1_lay1.append(layer1)
                self.rm1_lay2.append(layer2)
                n_prev = n_hid
                k = k + 1

            layer1 = nn.Linear(n_prev, 1)
            self.register_parameter(f'rm1_w1_{k}', layer1.weight)
            self.register_parameter(f'rm1_b1_{k}', layer1.bias)
            self.rm1_lay1.append(layer1)
        #############################################################

        #############################################################
        if len(config.rm2_hidden) == 0:
            if config.rm2 is not None:
                rm2 = nn.Parameter(torch.Tensor([config.rm2]), requires_grad=False)
            else:
                rm2 = nn.Parameter(torch.randn(1))
            self.register_parameter('rm2', rm2)
            self.rm2_lay1.append(rm2)
        else:
            n_prev = self.input
            k = 1
            for n_hid, act in zip(config.rm2_hidden, config.rm2_act):
                layer1 = nn.Linear(n_prev, n_hid)

                if act == 'SKIP':
                    layer2 = nn.Linear(self.input, n_hid)
                else:
                    layer2 = nn.Linear(n_prev, n_hid)

                self.register_parameter(f'rm2_w1_{k}', layer1.weight)
                self.register_parameter(f'rm2_w2_{k}', layer2.weight)

                self.register_parameter(f'rm2_b1_{k}', layer1.bias)
                self.register_parameter(f'rm2_b2_{k}', layer2.bias)

                self.rm2_lay1.append(layer1)
                self.rm2_lay2.append(layer2)
                n_prev = n_hid
                k = k + 1

            layer1 = nn.Linear(n_prev, 1)
            self.register_parameter(f'rm2_w1_{k}', layer1.weight)
            self.register_parameter(f'rm2_b1_{k}', layer1.bias)
            self.rm2_lay1.append(layer1)
        #############################################################

    def forward(self, data):
        l1, l2, I, U, g1, g2, l1_dot, l2_dot = data
        #############################################################
        # loss 1
        b1_y = self.net_out(I, self.config.b1_act, self.b1_lay1, self.b1_lay2)
        #############################################################
        # loss 2
        bl_1 = self.net_out(l1, self.config.b1_act, self.b1_lay1, self.b1_lay2)
        b1_grad = self.get_gradient(l1, l1_dot, self.config.b1_act, self.b1_lay1, self.b1_lay2)
        if len(self.config.bm1_hidden) == 0:
            bm1_y = l1 * 0 + self.bm1_lay1[0]
        else:
            bm1_y = self.net_out(l1, self.config.bm1_act, self.bm1_lay1, self.bm1_lay2)
        #############################################################
        # loss 3
        bl_2 = self.net_out(l2, self.config.b2_act, self.b2_lay1, self.b2_lay2)
        b2_grad = self.get_gradient(l2, l2_dot, self.config.b2_act, self.b2_lay1, self.b2_lay2)
        if len(self.config.bm2_hidden) == 0:
            bm2_y = l2 * 0 + self.bm2_lay1[0]
        else:
            bm2_y = self.net_out(l2, self.config.bm2_act, self.bm2_lay1, self.bm2_lay2)
        #############################################################
        # loss 4
        g1_tran = self.transform_data(g1, self.config.example.r1)
        b_l2_y = self.net_out(g1_tran, self.config.b2_act, self.b2_lay1, self.b2_lay2)
        b_l1_y = self.net_out(g1, self.config.b1_act, self.b1_lay1, self.b1_lay2)
        #############################################################
        # loss 5
        g2_tran = self.transform_data(g2, self.config.example.r2)
        bb_l1_y = self.net_out(g2_tran, self.config.b1_act, self.b1_lay1, self.b1_lay2)
        bb_l2_y = self.net_out(g2, self.config.b2_act, self.b2_lay1, self.b2_lay2)
        #############################################################
        # loss 6
        if len(self.config.rm1_hidden) == 0:
            rm1_y = g1 * 0 + self.rm1_lay1[0]
        else:
            rm1_y = self.net_out(g1, self.config.rm1_act, self.rm1_lay1, self.rm1_lay2)
        #############################################################
        # loss 7
        if len(self.config.rm2_hidden) == 0:
            rm2_y = g2 * 0 + self.rm2_lay1[0]
        else:
            rm2_y = self.net_out(g2, self.config.rm2_act, self.rm2_lay1, self.rm2_lay2)
        #############################################################
        # loss 8
        b2_y = self.net_out(U, self.config.b2_act, self.b2_lay1, self.b2_lay2)

        return b1_y, bl_1, b1_grad, bm1_y, bl_2, b2_grad, bm2_y, b_l2_y, b_l1_y, bb_l1_y, bb_l2_y, rm1_y, rm2_y, b2_y

    def net_out(self, x, act, lay1, lay2):
        y = x
        for idx, (layer1, layer2) in enumerate(zip(lay1[:-1], lay2)):
            if act[idx] == 'SQUARE':
                z = layer1(y)
                y = z ** 2
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

    def transform_data(self, data, f):
        ans = [torch.unsqueeze(torch.tensor(list(map(ff, data))), dim=1) for ff in f]
        return torch.cat(ans, dim=1)

    def get_gradient(self, x, xdot, act, lay1, lay2):
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
        expr_1 = self.sp_net(x, self.config.b1_act, self.b1_lay1, self.b1_lay2)
        expr.append(expr_1)
        #############################################################
        expr_2 = self.sp_net(x, self.config.b2_act, self.b2_lay1, self.b2_lay2)
        expr.append(expr_2)
        #############################################################
        if len(self.config.bm1_hidden) == 0:
            expr_3 = self.bm1_lay1[0].detach().numpy()[0]
        else:
            expr_3 = self.sp_net(x, self.config.bm1_act, self.bm1_lay1, self.bm1_lay2)
        expr.append(expr_3)
        #############################################################
        if len(self.config.bm2_hidden) == 0:
            expr_4 = self.bm2_lay1[0].detach().numpy()[0]
        else:
            expr_4 = self.sp_net(x, self.config.bm2_act, self.bm2_lay1, self.bm2_lay2)
        expr.append(expr_4)
        #############################################################
        if len(self.config.rm1_hidden) == 0:
            expr_5 = self.rm1_lay1[0].detach().numpy()[0]
        else:
            expr_5 = self.sp_net(x, self.config.rm1_act, self.rm1_lay1, self.rm1_lay2)
        expr.append(expr_5)
        #############################################################
        if len(self.config.rm2_hidden) == 0:
            expr_6 = self.rm2_lay1[0].detach().numpy()[0]
        else:
            expr_6 = self.sp_net(x, self.config.rm2_act, self.rm2_lay1, self.rm2_lay2)
        expr.append(expr_6)

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
