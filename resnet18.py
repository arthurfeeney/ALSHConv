
import torch.nn
import torch.nn as nn
import torch.nn.functional as F
from Hash.stable_distribution import StableDistribution
from ALSHLayers.alsh_linear import ALSHLinear
from ALSHLayers.alsh_conv2d import ALSHConv2d
from Hash.norm_and_halves import append_norm_powers, append_halves
from Utility.cp_utils import zero_fill_missing
from simp_conv2d import SimpConv2d

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding, same=True, bias=False):
        super(Block, self).__init__()

        self.in_channels = in_channels,
        self.out_channels = out_channels,
        self.kernel_size = kernel_size,
        self.stride = stride
        self.padding = padding
        self.bias = bias

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size,
                                stride, padding, bias=bias)
        self.bn1   = nn.BatchNorm2d(out_channels)
        if same:
            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size,
                                    stride, padding, bias=bias)
        else:
            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size,
                                    stride*2, padding, bias=bias)
        self.bn2   = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.bn2(x)
        return x


class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()


        # 224x224 -> 112x112
        self.conv1 = nn.Conv2d(3, 64, 7, 2, 1)
        self.pool1 = nn.MaxPool2d(3, stride=2)

        # 112x112 -> 56x56
        self.block1 = Block(64, 64, 3, 1, 1)
        self.block2 = Block(64, 64, 3, 1, 1, same=False)

        # 56x56 -> 28x28
        self.block3 = Block(64, 128, 3, 1, 1)
        self.block4 = Block(128, 128, 3, 1, 1, same=False)

        # 28x28 -> 14x14
        self.block5 = Block(128, 256, 3, 1, 1)
        self.block6 = Block(256, 256, 3, 1, 1, same=False)

        # 14x14 -> 7x7
        self.block7 = Block(256, 512, 3, 1, 1)
        self.block8 = Block(512, 512, 3, 1, 1, same=False)

        #7x7 -> 1x1
        self.pool2 = nn.AvgPool2d(2)

        self.fc = nn.Linear(2048, 1000)


    def forward(self, x):
        r"""

        """
        batch_size = x.size()[0]
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool1(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.pool2(x)
        x = x.view(batch_size, -1)
        x = self.fc(x)

        return x
