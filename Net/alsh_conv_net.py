
import torch.nn
import torch.nn as nn
import torch.nn.functional as F
from Hash.stable_distribution import StableDistribution
from ALSHLayers.alsh_linear import ALSHLinear
from ALSHLayers.alsh_conv2d import ALSHConv2d
from ALSHLayers.F_alsh_conv2d import F_ALSHConv2d
from Hash.norm_and_halves import append_norm_powers, append_halves
from Utility.cp_utils import zero_fill_missing
from simp_conv2d import SimpConv2d

from Hash.sign_random_projection import SignRandomProjection
from Hash.sub_norm_and_zeros import append_sub_norm_powers, append_zeros

class ALSHConvNet(nn.Module):
    def __init__(self, device=torch.device('cuda')):
        super(ALSHConvNet, self).__init__()

        self.device=device

        self.l1 = F_ALSHConv2d(3, 16, 5, 1, 2, 1, False, 6, 
                               SignRandomProjection, 2,
                               25, .9999, P=append_sub_norm_powers,
                               Q=append_zeros, device=device)

        #self.l1 = nn.Conv2d(3, 16, 5, 1, 2, bias=False)
        self.p1 = nn.MaxPool2d(2)
        self.l2 = F_ALSHConv2d(16, 20, 5, 1, 2, 1, False, 6, 
                               SignRandomProjection, 2,
                               25, .9999, P=append_sub_norm_powers,
                               Q=append_zeros, device=device)

        #self.l2 = nn.Conv2d(16, 20, 5, 1, 2, bias=False)
        self.p2 = nn.MaxPool2d(2)
        self.l3 = F_ALSHConv2d(20, 20, 5, 1, 2, 1, False, 10, 
                               SignRandomProjection, 3,
                               25, .9999, P=append_sub_norm_powers,
                               Q=append_zeros, device=device)
        #self.l3 = nn.Conv2d(20, 20, 5, 1, 2, bias=False)
        self.p3 = nn.MaxPool2d(2)
        self.out = nn.Linear(320, 10)

    def forward(self, x, mode=True):
        batch_size = x.size()[0]
        x, i= self.l1(x)
        #x    = self.l1(x)
        x    = F.relu(x)
        x    = self.p1(x)
        x, i = self.l2(x, i)
        #x    = self.l2(x)
        x    = F.relu(x)
        x    = self.p2(x)
        x, i = self.l3(x, i)
        #x    = self.l3(x)
        x    = F.relu(x)

        x = zero_fill_missing(x, i, (batch_size, 20, 8, 8), self.device)
        x = self.p3(x)

        x = x.view(batch_size, -1)

        x = self.out(x)
        return x
