
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


class NormVGGNet(nn.Module):
    def __init__(self, device=torch.device('cuda')):
        super(NormVGGNet, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1, 1, bias=False)
        self.conv2 = nn.Conv2d(64, 64, 3, 1, 1, 1, bias=False)
        self.pool1 = nn.MaxPool2d(2)

        self.conv3 = nn.Conv2d(64, 128, 3, 1, 1, 1, bias=False)
        self.conv4 = nn.Conv2d(128, 128, 3, 1, 1, 1, bias=False)
        self.pool2 = nn.MaxPool2d(2)

        self.conv5 = nn.Conv2d(128, 256, 3, 1, 1, 1, bias=False)
        self.conv6 = nn.Conv2d(256, 256, 3, 1, 1, 1, bias=False)
        self.pool3 = nn.MaxPool2d(2)

        self.fc7   = nn.Linear(4096, 512)
        self.fc8 = nn.Linear(512, 512)

        self.fc9 = nn.Linear(512, 10)

    def forward(self, x, mode=True):
        batch_size = x.size()[0]
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool1(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = self.pool2(x)
        x = F.relu(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = F.relu(x)
        x = self.pool3(x)
        x = x.view(batch_size, -1) # flatten each thing in batch
        x = self.fc7(x)
        x = F.relu(x)
        x = self.fc8(x)
        x = F.relu(x)


        x = self.fc9(x)

        return x
