import numpy as np
import torch.nn as nn
import torch

class NMELoss(nn.Module):
    def __init__(self):
        super(NMELoss, self).__init__()

    def forward(self, x, y):
        dis = x - y
        dis = torch.sqrt(torch.sum(torch.pow(dis, 2), 2))
        dis = torch.mean(dis) / 384

        return dis


class WingLoss(nn.Module):
    def __init__(self, gamma=10, eps=2):
        super(WingLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps

    def forward(self, x, y):

        # (N, 68, 2)
        dis = x - y
        dis = torch.sqrt(torch.sum(torch.pow(dis, 2), 2))
        dis = torch.mean(dis)
        if dis < self.eps:
            loss = self.gamma * torch.log(1 + dis/self.eps)
        else:
            loss = dis - (self.gamma - self.gamma * np.log(1 + self.gamma/self.eps))

        return loss
        