
import torch.nn
import torch.nn as nn
import torch.nn.functional as F
from Hash.stable_distribution import StableDistribution
from ALSHLayers.alsh_linear import ALSHLinear
from ALSHLayers.alsh_conv2d import ALSHConv2d
from Hash.norm_and_halves import append_norm_powers, append_halves
from Utility.cp_utils import zero_fill_missing
from simp_conv2d import SimpConv2d


class ALSHVGGNet(nn.Module):
    def __init__(self):
        super(ALSHVGGNet, self).__init__()

        # the first argument is just kkc + m for whichever conv its used with
        self.h1 = StableDistribution(37,  .002)
        self.h2 = StableDistribution(586,  .002)
        self.h3 = StableDistribution(586,  .002)
        self.h4 = StableDistribution(1162,  .002)
        self.h5 = StableDistribution(1162,  .002)
        self.h6 = StableDistribution(2314,  .002)
        self.h7 = StableDistribution(2314,  .002)
        self.h8 = StableDistribution(4618,  .002)


        self.conv1 = SimpConv2d(3, 64, 3, 1, 1, False)#ALSHConv2d(3, 64, 3, 1, 1, True, self.h1, 1, 10,
                     #           P=append_norm_powers, Q=append_halves)
        self.conv2 = SimpConv2d(64, 64, 3, 1, 1, False)#ALSHConv2d(64, 64, 3, 1, 1, True, self.h2, 1, 10,
                     #           P=append_norm_powers, Q=append_halves)
        self.pool1 = nn.MaxPool2d(2)

        self.conv3 = SimpConv2d(64, 128, 3, 1, 1, False)#ALSHConv2d(64, 128, 3, 1, 1, True, self.h3, 1, 10,
                     #           P=append_norm_powers, Q=append_halves)
        self.conv4 = SimpConv2d(128, 128, 3, 1, 1, False)#ALSHConv2d(128, 128, 3, 1, 1, True, self.h4, 1, 10,
                     #           P=append_norm_powers, Q=append_halves)
        self.pool2 = nn.MaxPool2d(2)

        self.conv5 = SimpConv2d(128, 256, 3, 1, 1, False)#ALSHConv2d(128, 256, 3, 1, 1, True, self.h5, 2, 10,
                              #P=append_norm_powers, Q=append_halves)
        self.conv6 = ALSHConv2d(256, 256, 3, 1, 1, True, self.h6, 2, 10,
                                P=append_norm_powers, Q=append_halves)
        self.pool3 = nn.MaxPool2d(2)

        self.fc7   = nn.Linear(4096, 512)
        self.fc8 = nn.Linear(512, 512)

        self.fc9 = nn.Linear(512, 10)

    def forward(self, x, mode=True):
        batch_size = x.size()[0]
        x    = self.conv1(x)
        x    = F.relu(x)
        x    = self.conv2(x)
        x    = F.relu(x)
        x    = self.pool1(x)
        x    = self.conv3(x)
        x    = F.relu(x)
        x    = self.conv4(x)
        x    = self.pool2(x)
        x    = F.relu(x)
        x  = self.conv5(x)
        x, i= self.conv6(x, mode)
        x    = F.relu(x)
        x    = self.pool3(x)
        x    = zero_fill_missing(x, i, (batch_size, 256, 4, 4))
        x    = x.view(batch_size, -1)
        x    = self.fc7(x)
        x    = F.relu(x)
        x    = self.fc8(x)
        x    = F.relu(x)


        x = self.fc9(x)

        return x
