
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

        self.h1 = StableDistribution(27 + 5, .1)
        self.h2 = StableDistribution(45 + 5, .1)

        self.l1 = ALSHConv2d(3, 5, 3, 1, 0, 0, self.h1, 2, 5,
                             P=append_norm_powers, Q=append_halves)
        self.l2 = ALSHConv2d(5, 5, 3, 1, 0, 0, self.h2, 2, 5,
                            P=append_norm_powers, Q=append_halves)
        self.out = nn.Linear(1000, 10)

    def forward(self, x, mode=True):
        x = F.relu(self.l1(x, mode))
        x = F.relu(self.l2(x, mode))
        print(x.size())
        x = self.out(x)
        return x
