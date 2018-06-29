
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


class ALSHVGGNet(nn.Module):
    def __init__(self, device=torch.device('cuda')):
        super(ALSHVGGNet, self).__init__()

        self.device=device

        # the first argument is just kkc + m for whichever conv its used with
        self.h1 = StableDistribution(36,  .002, device=device)
        self.h2 = StableDistribution(585,  .002, device=device)
        self.h3 = StableDistribution(585,  .002, device=device)
        self.h4 = StableDistribution(1161,  .002, device=device)
        self.h5 = StableDistribution(1161,  .002, device=device)
        self.h6 = StableDistribution(2313,  .002, device=device)
        self.h7 = StableDistribution(2313,  .002, device=device)
        self.h8 = StableDistribution(4617,  .002, device=device)


        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1, 1, bias=False)#F_ALSHConv2d(3, 64, 3, 1, 1, 1, False, self.h1, 5, 9,
                     #             P=append_norm_powers, Q=append_halves,
                     #             device=device)
        self.conv2 = nn.Conv2d(64, 64, 3, 1, 1, 1, bias=False)#F_ALSHConv2d(64, 64, 3, 1, 1, 1, False, self.h2, 5, 9,
                     #             P=append_norm_powers, Q=append_halves,
                     #             device=device)
        self.pool1 = nn.MaxPool2d(2)

        self.conv3 = nn.Conv2d(64, 128, 3, 1, 1, 1, bias=False)#F_ALSHConv2d(64, 128, 3, 1, 1, 1, False, self.h3, 5, 9,
                     #             P=append_norm_powers, Q=append_halves,
                     #             device=device)
        self.conv4 = nn.Conv2d(128, 128, 3, 1, 1, 1, bias=False)#F_ALSHConv2d(128, 128, 3, 1, 1, 1, False, self.h4, 5, 9,
                     #             P=append_norm_powers, Q=append_halves,
                     #             device=device)
        self.pool2 = nn.MaxPool2d(2)

        self.conv5 = nn.Conv2d(128, 256, 3, 1, 1, 1, bias=False)#F_ALSHConv2d(128, 256, 3, 1, 1, 1, False, self.h5, 2, 9,
                                  #P=append_norm_powers, Q=append_halves,
                                  #device=device)
        self.conv6 = F_ALSHConv2d(256, 256, 3, 1, 1, 1, False, self.h6, 2, 9,
                                  P=append_norm_powers, Q=append_halves,
                                  device=device)
        self.pool3 = nn.MaxPool2d(2)

        self.fc7   = nn.Linear(4096, 512)
        self.fc8 = nn.Linear(512, 512)

        self.fc9 = nn.Linear(512, 10)

    def forward(self, x, mode=True):
        batch_size = x.size()[0]
        #x, i = self.conv1(x)
        x    = self.conv1(x)
        x    = F.relu(x)
        #x, i = self.conv2(x, i)
        x    = self.conv2(x)
        x    = F.relu(x)
        x    = self.pool1(x)
        #x, i = self.conv3(x, i)
        x    = self.conv3(x)
        x    = F.relu(x)
        #x, i = self.conv4(x, i)
        x    = self.conv4(x)
        x    = self.pool2(x)
        x    = F.relu(x)
        #x, i = self.conv5(x)
        x    = self.conv5(x)
        x, i = self.conv6(x)
        x    = F.relu(x)
        x    = self.pool3(x)
        x    = zero_fill_missing(x, i, (batch_size, 256, 4, 4),
                                 device=self.device)
        x    = x.view(batch_size, -1)
        x    = self.fc7(x)
        x    = F.relu(x)
        x    = self.fc8(x)
        x    = F.relu(x)


        x = self.fc9(x)

        return x
