
import torch.nn
import torch.nn as nn
import torch.nn.functional as F
from Hash.stable_distribution import StableDistribution
from ALSHLayers.alsh_linear import ALSHLinear
from ALSHLayers.alsh_conv2d import ALSHConv2d
from Hash.norm_and_halves import append_norm_powers, append_halves
from cp_utils import zero_fill_missing


class ALSHConvNet(nn.Module):
    def __init__(self):
        super(ALSHConvNet, self).__init__()

        self.h1 = StableDistribution(75 + 5, .2)
        self.h2 = StableDistribution(400 + 5, .2)
        self.h3 = StableDistribution(500 + 5, .2)

        self.l1 = ALSHConv2d(3, 16, 5, 1, 2, True, self.h1, 2, 5,
                             P=append_norm_powers, Q=append_halves)
        #self.l1 = nn.Conv2d(3, 16, 5, 1, 2, bias=False)
        self.p1 = nn.MaxPool2d(2)
        self.l2 = ALSHConv2d(16, 20, 5, 1, 2, True, self.h2, 2, 5,
                            P=append_norm_powers, Q=append_halves)
        #self.l2 = nn.Conv2d(16, 20, 5, 1, 2, bias=False)
        self.p2 = nn.MaxPool2d(2)
        self.l3 = ALSHConv2d(20, 20, 5, 1, 2, True, self.h3, 2, 5,
                             P=append_norm_powers, Q=append_halves)
        #self.l3 = nn.Conv2d(20, 20, 5, 1, 2, bias=False)
        self.p3 = nn.MaxPool2d(2)
        self.out = nn.Linear(320, 10)

    def forward(self, x, mode=True):
        batch_size = x.size()[0]
        x, i = self.l1(x, mode)
        x    = F.relu(x)
        x    = self.p1(x)
        x, i = self.l2(x, mode, i)
        x    = F.relu(x)
        x    = self.p2(x)
        x, i = self.l3(x, mode, i)
        x    = F.relu(x)

        t = zero_fill_missing(x, i, (batch_size, 20, 8, 8))
        x = self.p3(t)

        x = x.view(batch_size, -1)

        x = self.out(x)
        return x
