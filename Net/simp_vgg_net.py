
import torch.nn
import torch.nn as nn
import torch.nn.functional as F
from Hash.stable_distribution import StableDistribution
from ALSHLayers.alsh_linear import ALSHLinear
from ALSHLayers.alsh_conv2d import ALSHConv2d
from Hash.norm_and_halves import append_norm_powers, append_halves
from Utility.cp_utils import zero_fill_missing
from simp_conv2d import SimpConv2d

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 bias=False):
        super(ResBlock, self).__init__()
        self.conv1 = SimpConv2d(in_channels, out_channels, kernel_size,
                                stride, 0, bias)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.Relu()

        self.conv2 = SimpConv2d(out_channels, out_channels, kernel_size,
                                stride, 0, bias)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.Relu()

    def forward(self, x):
        y = self.conv1(x)
        y = self.bn1(y)
        y = self.relu1(y)
        y = self.conv2(y)
        y = self.bn2(y)
        y = self.relu2(y)
        return y + x



class SimpResNet_18(nn.Module):
    def __init__(self):
        super(SimpResNet_18, self).__init__()


        self.conv1 = SimpConv2d(3, 64, 3, 1, 1, False)


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
        x    = self.conv5(x)
        x    = self.conv6(x)
        x    = F.relu(x)
        x    = self.pool3(x)
        x    = x.view(batch_size, -1)
        x    = self.fc7(x)
        x    = F.relu(x)
        x    = self.fc8(x)
        x    = F.relu(x)

        x = self.fc9(x)

        return x
