
import torch.nn
import torch.nn as nn
import torch.nn.functional as F
from Hash.stable_distribution import StableDistribution
from ALSHLayers.alsh_linear import ALSHLinear
from ALSHLayers.alsh_conv2d import ALSHConv2d
from Hash.norm_and_halves import append_norm_powers, append_halves


class ALSHConvNet(nn.Module):
    def __init__(self):
        super(ALSHConvNet, self).__init__()

        self.h1 = StableDistribution(75 + 5, .1)
        self.h2 = StableDistribution(400 + 5, .1)
        self.h3 = StableDistribution(500 + 5, .1)

        #self.l1 = ALSHConv2d(3, 16, 5, 1, 2, 0, self.h1, 2, 5,
        #                     P=append_norm_powers, Q=append_halves)
        self.l1 = nn.Conv2d(3, 16, 5, 1, 2)
        self.p1 = nn.MaxPool2d(2)
        #self.l2 = ALSHConv2d(16, 20, 5, 1, 2, 0, self.h2, 2, 5,
        #                    P=append_norm_powers, Q=append_halves)
        self.l2 = nn.Conv2d(16, 20, 5, 1, 2)
        self.p2 = nn.MaxPool2d(2)
        #self.l3 = ALSHConv2d(20, 20, 5, 1, 2, 0, self.h3, 2, 5,
        #                     P=append_norm_powers, Q=append_halves)
        self.l3 = nn.Conv2d(20, 20, 5, 1, 2)
        self.p3 = nn.MaxPool2d(2)
        self.out = nn.Linear(320, 10)

    def forward(self, x, mode=True):
        batch_size = x.size()[0]
        x = F.relu(self.l1(x))
        x = self.p1(x)
        x = F.relu(self.l2(x))
        x = self.p2(x)
        x = F.relu(self.l3(x))
        x = self.p3(x)
        x = x.view(batch_size, -1).cuda() # flatten
        x = self.out(x)
        return x
