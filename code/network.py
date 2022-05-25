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