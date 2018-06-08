
import torch.nn
import torch.nn as nn
import torch.nn.functional as F
from Hash.stable_distribution import StableDistribution
from ALSHLayers.alsh_linear import ALSHLinear
from ALSHLayers.alsh_conv2d import ALSHConv2d
from Hash.norm_and_halves import append_norm_powers, append_halves
from cp_utils import zero_fill_missing


class ALSHVGGNet(nn.Module):
    def __init__(self):
        super(ALSHAlexNet, self).__init__()

        # the first argument is just kkc + m for whichever conv its used with
        self.h1 = StableDistribution(368,  2)

        self.conv1 = ALSHConv2d(3, 64, 3, 1, 0, True, self.h1, 6, 5,
                                P=append_norm_powers, Q=append_halves)
        self.conv2 = ALSHConv2d(64, 64, 3, 1, 0, True, self.h2, 6, 5,
                                P=append_norm_powers, Q=append_halves)
        self.pool1 = nn.MaxPool2d(2)

        self.conv3 = ALSHConv2d(64, 128, 3, 1, 0, True, self.h3, 6, 5,
                                P=append_norm_powers, Q=append_halves)
        self.conv4 = ALSHConv2d(128, 128, 3, 1, 0, True, self.h4, 6, 5,
                                P=append_norm_powers, Q=append_halves)
        self.pool2 = nn.MaxPool2d(2)

        self.conv5 = ALSHConv2d(128, 256, 3, 1, 0, True, self.h5, 6, 5,
                              P=append_norm_powers, Q=append_halves)
        self.conv6 = ALSHConv2d(256, 256, 3, 1, 0, True, self.h6, 6, 5,
                                P=append_norm_powers, Q=append_halves)
        self.pool3 = nn.MaxPool2d(2)

        self.conv7 = ALSHConv2d(256, 512, 3, 1, 0, True, self.h7, 6, 5,
                              P=append_norm_powers, Q=append_halves)
        self.conv8 = ALSHConv2d(512, 512, 3, 1, 0, True, self.h8, 6, 5,
                                P=append_norm_powers, Q=append_halves)



        self.fc9 =

        self.fc6 = nn.Linear(256*6*6, 4096)

    def _input_hw(self, x):
        return x.size()[2:]

    def forward(self, x, mode=True):
        batch_size = x.size()[0]
        x, i = self.conv1(x, mode)

        t = zero_fill_missing(x, i, (batch_size, 256, 6, 6))

        return x
