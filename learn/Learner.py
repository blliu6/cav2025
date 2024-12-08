import torch
from utils.Config import CegisConfig


class Learner:
    def __init__(self, config: CegisConfig):
        if config.example.continuous:
            from learn.net_continuous import Net
            self.net = Net(config)
        else:
            from learn.net import Net
            self.net = Net(config)
        self.config = config

    def learn(self, data, opt):
        learn_loops = self.config.learning_loops
        margin = self.config.margin
        slope = 1e-3
        relu6 = torch.nn.ReLU6()
        optimizer = opt

        data_tensor = data

        for epoch in range(learn_loops):
            optimizer.zero_grad()

            b1_y, bl_1, b1_grad, bm1_y, bl_2, b2_grad, bm2_y, b_l2_y, b_l1_y, bb_l1_y, bb_l2_y, rm1_y, rm2_y, b2_y = self.net(
                data_tensor)

            b1_y, bl_1, bm1_y = b1_y[:, 0], bl_1[:, 0], bm1_y[:, 0]
            bl_2, bm2_y, b_l2_y = bl_2[:, 0], bm2_y[:, 0], b_l2_y[:, 0]
            b_l1_y, bb_l1_y, bb_l2_y = b_l1_y[:, 0], bb_l1_y[:, 0], bb_l2_y[:, 0]
            rm1_y, rm2_y, b2_y = rm1_y[:, 0], rm2_y[:, 0], b2_y[:, 0]

            weight = self.config.loss_weight
            accuracy = [0] * 8
            ###########
            # loss 1
            p = b1_y
            accuracy[0] = sum(p > margin / 2).item() * 100 / len(b1_y)

            loss_1 = weight[0] * (torch.relu(-p + margin) - slope * relu6(p - margin)).mean()
            ###########
            # loss 2
            p = b1_grad - bm1_y * bl_1
            accuracy[1] = sum(p > margin / 2).item() * 100 / len(bl_1)

            loss_2 = weight[1] * (torch.relu(-p + margin) - slope * relu6(p - margin)).mean()
            ###########
            # loss 3
            p = b2_grad - bm2_y * bl_2
            accuracy[2] = sum(p > margin / 2).item() * 100 / len(bl_2)

            loss_3 = weight[2] * (torch.relu(-p + margin) - slope * relu6(p - margin)).mean()
            ###########
            # loss 4
            p = b_l2_y - rm1_y * b_l1_y
            accuracy[3] = sum(p > margin / 2).item() * 100 / len(b_l1_y)

            loss_4 = weight[3] * (torch.relu(-p + margin) - slope * relu6(p - margin)).mean()
            ###########
            # loss 5
            p = bb_l1_y - rm2_y * bb_l2_y
            accuracy[4] = sum(p > margin / 2).item() * 100 / len(bb_l2_y)

            loss_5 = weight[4] * (torch.relu(-p + margin) - slope * relu6(p - margin)).mean()
            ###########
            # loss 6
            p = rm1_y
            accuracy[5] = sum(p > margin / 2).item() * 100 / len(rm1_y)

            loss_6 = weight[5] * (torch.relu(-p + margin) - slope * relu6(p - margin)).mean()
            ###########
            # loss 7
            p = rm2_y
            accuracy[6] = sum(p > margin / 2).item() * 100 / len(rm2_y)

            loss_7 = weight[6] * (torch.relu(-p + margin) - slope * relu6(p - margin)).mean()
            ###########
            # loss 8
            p = b2_y
            accuracy[7] = sum(p < -margin / 2).item() * 100 / len(b2_y)

            loss_8 = weight[7] * (torch.relu(p + margin) - slope * relu6(-p - margin)).mean()
            ###########
            loss = loss_1 + loss_2 + loss_3 + loss_4 + loss_5 + loss_6 + loss_7 + loss_8
            result = True
            for e in accuracy:
                result = result and (e == 100)

            if epoch % (learn_loops // 10) == 0 or result:
                print(f'{epoch}->', end=' ')
                for i in range(len(accuracy)):
                    print(f'accuracy{i + 1}:{accuracy[i]}', end=', ')
                print(f'loss:{loss}')

            loss.backward()
            optimizer.step()
            if result:
                break

    def learn_for_continous(self, data, opt):
        learn_loops = self.config.learning_loops
        margin = self.config.margin
        slope = 1e-3
        relu6 = torch.nn.ReLU6()
        optimizer = opt

        data_tensor = data

        for epoch in range(learn_loops):
            optimizer.zero_grad()

            b1_y, bl_1, b1_grad, bm1_y, b2_y = self.net(data_tensor)

            b1_y, bl_1, bm1_y, b2_y = b1_y[:, 0], bl_1[:, 0], bm1_y[:, 0], b2_y[:, 0]

            weight = self.config.loss_weight_continuous

            accuracy = [0] * 3
            ###########
            # loss 1
            p = b1_y
            accuracy[0] = sum(p > margin / 2).item() * 100 / len(b1_y)

            loss_1 = weight[0] * (torch.relu(-p + margin) - slope * relu6(p - margin)).mean()
            ###########
            # loss 2
            p = b1_grad - bm1_y * bl_1
            accuracy[1] = sum(p > margin / 2).item() * 100 / len(bl_1)

            loss_2 = weight[1] * (torch.relu(-p + margin) - slope * relu6(p - margin)).mean()
            ###########
            # loss 8
            p = b2_y
            accuracy[2] = sum(p < -margin / 2).item() * 100 / len(b2_y)

            loss_8 = weight[2] * (torch.relu(p + margin) - slope * relu6(-p - margin)).mean()
            ###########
            loss = loss_1 + loss_8 + loss_2
            result = True

            for e in accuracy:
                result = result and (e == 100)

            if epoch % (learn_loops // 10) == 0 or result:
                print(f'{epoch}->', end=' ')
                for i in range(len(accuracy)):
                    print(f'accuracy{i + 1}:{accuracy[i]}', end=', ')
                print(f'loss:{loss}')

            loss.backward()
            optimizer.step()
            if result:
                break
