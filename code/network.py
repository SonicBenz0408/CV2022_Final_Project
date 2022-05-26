from turtle import forward
import torch.nn as nn
import torch

class FacialLandmark(nn.Module):
    def __init__(self):
        super(FacialLandmark, self).__init__()
        
        # input: (3, 384, 384)

        self.dim = 32
        
        #384
        self.input_conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=self.dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(self.dim),
            nn.ReLU(inplace=True)
        )
        #192
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=self.dim, out_channels=self.dim*2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(self.dim*2),
            nn.ReLU(inplace=True)
        )
        #96
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=self.dim*2, out_channels=self.dim*2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(self.dim*2),
            nn.ReLU(inplace=True)
        )
        #48
        self.pool1 = nn.MaxPool2d(2, 2)
        
        #24
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=self.dim*2, out_channels=self.dim*4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(self.dim*4),
            nn.ReLU(inplace=True)
        )
        #12
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=self.dim*4, out_channels=self.dim*8, kernel_size=3, stride=2, padding=1),
            #nn.BatchNorm2d(self.dim*8),
            #nn.ReLU(inplace=True)
        )

        # (N, 256, 6, 6)

        self.fc = nn.Sequential(
            nn.Linear(256 * 6 * 6, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 2 * 68)
        )

    def forward(self, input):
        x = self.input_conv(input)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool1(x)

        x = self.conv3(x)
        x = self.conv4(x)

        x = x.view(-1, 256 * 6 * 6)

        x = self.fc(x)
        
        out = torch.reshape(x, (x.shape[0], 68, 2))

        return out

class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.InstanceNorm2d(oup, affine=True)
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.InstanceNorm2d(oup, affine=True)
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class PyramidNet(nn.Module):

    def __init__(self):
        super(PyramidNet, self).__init__()

        self.dim = 32

        # expand, out_channels, layer_num, stride
        self.mobile_config = [
            [1, 16, 1, 1],
            [2, 32, 1, 2],
            [2, 32, 5, 2],
            [2, 64, 1, 2],
            [4, 128, 6, 2],
            [2, 16, 1, 1]
        ]
        self.input_conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=self.dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.dim),
            nn.ReLU6(inplace=True)
        )

        self.feature_extractor = [self.input_conv]
        
        input_channel = self.dim

        for t, c, n, s in self.mobile_config:
            output_channel = c
            for i in range(n):
                if i == 0:
                    self.feature_extractor.append(InvertedResidual(input_channel, output_channel, s, expand_ratio=t))
                else:
                    self.feature_extractor.append(InvertedResidual(input_channel, output_channel, 1, expand_ratio=t))
                input_channel = output_channel

        self.feature_extractor = nn.Sequential(*self.feature_extractor)

        # (N, 16, 12, 12)
        self.down1 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1)
        
        # (N, 32, 6, 6)
        self.down2 = nn.Sequential(
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=128, kernel_size=6, stride=1)
        )

        # (N, 128, 1, 1)
        # 16 * 12 * 12 + 32 * 6 * 6 + 128
        self.fc = nn.Linear(16 * 12 * 12 + 32 * 6 * 6 + 128, 2 * 68)


    def forward(self, input):
        s1 = self.feature_extractor(input)
        s2 = self.down1(s1)
        s3 = self.down2(s2)

        s1 = torch.flatten(s1, 1)
        s2 = torch.flatten(s2, 1)
        s3 = torch.flatten(s3, 1)

        feature = torch.concat((s1, s2, s3), dim=1)
        output = self.fc(feature)
        output = torch.reshape(output, (output.shape[0], 68, 2))

        return output