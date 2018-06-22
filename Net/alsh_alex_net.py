
import torch.nn
import torch.nn as nn
import torch.nn.functional as F
from Hash.stable_distribution import StableDistribution
from ALSHLayers.alsh_linear import ALSHLinear
from ALSHLayers.alsh_conv2d import ALSHConv2d
from Hash.norm_and_halves import append_norm_powers, append_halves
from Utility.cp_utils import zero_fill_missing


class ALSHAlexNet(nn.Module):
    def __init__(self):
        super(ALSHAlexNet, self).__init__()

        # the first argument is just kkc + m for whichever conv its used with
        self.h1 = StableDistribution(368,  2)
        self.h2 = StableDistribution(2405, 2)
        self.h3 = StableDistribution(2309, 2)
        self.h4 = StableDistribution(3461, 2)
        self.h5 = StableDistribution(3461, 2)

        self.conv1 = ALSHConv2d(3, 96, 11, 4, 0, True, self.h1, 6, 5,
                                P=append_norm_powers, Q=append_halves)
        self.p1 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv2 = ALSHConv2d(96, 256, 5, 1, 2, True, self.h2, 6, 5,
                               P=append_norm_powers, Q=append_halves)
        self.p2 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv3 = ALSHConv2d(256, 384, 3, 1, 1, True, self.h3, 6, 5,
                                P=append_norm_powers, Q=append_halves)

        self.conv4 = ALSHConv2d(384, 384, 3, 1, 1, True, self.h4, 6, 5,
                                P=append_norm_powers, Q=append_halves)

        self.conv5 = ALSHConv2d(384, 256, 3, 1, 1, True, self.h5, 6, 5,
                                P=append_norm_powers, Q=append_halves)
        self.p3 = nn.MaxPool2d(kernel_size=3, stride=2)

        # output of p3 is 256 * 3 * 3!
        self.fc6 = nn.Linear(256*6*6, 4096)
        self.fc7 = nn.Linear(4096, 4096)
        self.fc8 = nn.Linear(4096, 1000)

    def _input_hw(self, x):
        return x.size()[2:]

    def forward(self, x, mode=True):
        r"""
        x really needs to be N x 3 x 227 x 277   !
        """

        #assert self._input_hw(x) == [227, 227], "input is the wrong dim!"

        batch_size = x.size()[0]
        x, i = self.conv1(x, mode)
        x    = self.p1(x)
        x, i = self.conv2(x, mode, i)
        x    = self.p2(x)
        x, i = self.conv3(x, mode, i)
        x, i = self.conv4(x, mode, i)
        x, i = self.conv5(x, mode, i)
        x    = self.p3(x)

        t = zero_fill_missing(x, i, (batch_size, 256, 6, 6))

        x = t.view(batch_size, -1)

        x = self.fc6(x)
        x = self.fc7(x)
        x = self.fc8(x)

        return x
