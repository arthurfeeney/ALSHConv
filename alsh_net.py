
import torch.nn
import torch.nn as nn
import torch.nn.functional as F
from Hash.stable_distribution import StableDistribution
from ALSHLayers.alsh_linear import ALSHLinear
from ALSHLayers.alsh_conv2d import ALSHConv2d
from Hash.norm_and_halves import append_norm_powers, append_halves


class ALSHNet(nn.Module):
    def __init__(self):
        super(ALSHNet, self).__init__()
        self.l1 = ALSHLinear(3072, 1000, StableDistribution(3072+5, .1), 10, 5,
                            P=append_norm_powers, Q=append_halves)
        self.l2 = ALSHLinear(1000, 1000, StableDistribution(1000+5, .1), 10, 5,
                            P=append_norm_powers, Q=append_halves)
        self.out = nn.Linear(1000, 10)

    def forward(self, x, mode=True):
        x = F.relu(self.l1(x, mode))
        x = F.relu(self.l2(x, mode))
        x = self.out(x)
        return x
