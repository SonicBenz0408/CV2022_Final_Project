import numpy as np
import torch.nn as nn
import torch

class NMELoss(nn.Module):
    def __init__(self):
        super(NMELoss, self).__init__()

    def forward(self, x, y):
        dis = x - y
        dis = torch.sqrt(torch.sum(torch.pow(dis, 2), 2))
        dis = torch.mean(dis, 1)
        dis = torch.sum(dis) / 384

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

        small = torch.sum(self.gamma * torch.log(1 + dis[dis < self.gamma]/self.eps))
        large = torch.sum(dis[dis >= self.gamma] - (self.gamma - self.gamma * np.log(1 + self.gamma/self.eps)))

        loss = small + large

        return loss

class WeightedL2Loss(nn.Module):
    def __init__(self, weights):
        super(WeightedL2Loss, self).__init__()

        self.weights = weights
    
    def forward(self, x, y):

        dis = x - y
        dis = torch.sqrt(torch.sum(torch.pow(dis, 2), 2))
        dis = self.weights.repeat(dis.shape[0], 1).cuda() * dis
        dis = torch.mean(dis, 1)
        dis = torch.sum(dis)

        return dis

# right face: 1~9
# left face: 10~17
# right eyebrow: 18~22
# left eyebrow: 23~27
# nose: 28~31 32~36
# right eye: 37~42
# left eye: 43~48
# outer mouth: 49~60
# inner mouth: 61~68

class CenterLoss(nn.Module):
    def __init__(self):
        super(CenterLoss, self).__init__()
    
    def forward(self, x, y):
        # (N, 68, 2)
        face = torch.sum((torch.mean(x[:, 0:9, 0], 1) - torch.mean(y[:, 0:9, 0], 1)) ** 2 + (torch.mean(x[:, 0:9, 1], 1) - torch.mean(y[:, 0:9, 1], 1)) ** 2) + torch.sum((torch.mean(x[:, 9:17, 0], 1) - torch.mean(y[:, 9:17, 0], 1)) ** 2 + (torch.mean(x[:, 9:17, 1], 1) - torch.mean(y[:, 9:17, 1], 1)) ** 2)
        eyebrow = torch.sum((torch.mean(x[:, 17:22, 0], 1) - torch.mean(y[:, 17:22, 0], 1)) ** 2 + (torch.mean(x[:, 17:22, 1], 1) - torch.mean(y[:, 17:22, 1], 1)) ** 2) + torch.sum((torch.mean(x[:, 22:27, 0], 1) - torch.mean(y[:, 22:27, 0], 1)) ** 2 + (torch.mean(x[:, 22:27, 1], 1) - torch.mean(y[:, 22:27, 1], 1)) ** 2)
        nose = torch.sum((torch.mean(x[:, 27:36, 0], 1) - torch.mean(y[:, 27:36, 0], 1)) ** 2 + (torch.mean(x[:, 27:36, 1], 1) - torch.mean(y[:, 27:36, 1], 1)) ** 2)
        eye = torch.sum((torch.mean(x[:, 36:42, 0], 1) - torch.mean(y[:, 36:42, 0], 1)) ** 2 + (torch.mean(x[:, 36:42, 1], 1) - torch.mean(y[:, 36:42, 1], 1)) ** 2) + torch.sum((torch.mean(x[:, 42:48, 0], 1) - torch.mean(y[:, 42:48, 0], 1)) ** 2 + (torch.mean(x[:, 42:48, 1], 1) - torch.mean(y[:, 42:48, 1], 1)) ** 2)
        mouth = torch.sum((torch.mean(x[:, 48:60, 0], 1) - torch.mean(y[:, 48:60, 0], 1)) ** 2 + (torch.mean(x[:, 48:60, 1], 1) - torch.mean(y[:, 48:60, 1], 1)) ** 2) + torch.sum((torch.mean(x[:, 60:68, 0], 1) - torch.mean(y[:, 60:68, 0], 1)) ** 2 + (torch.mean(x[:, 60:68, 1], 1) - torch.mean(y[:, 60:68, 1], 1)) ** 2)
        
        dis = ( face + eyebrow + nose + eye + mouth ) / x.shape[0]
        return dis