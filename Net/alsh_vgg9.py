import torch.nn
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

import Net.quality_utils as QU 

from Conv.alsh_conv_2d import ALSHConv2d

from LSH.multi_hash_srp import MultiHash_SRP

def zero_fill_missing(x, i, dims, device=torch.device('cuda')):
    r"""
    fills channels that weren't computed with zeros.
    """
    if i is None:
        return x
    t = torch.empty(dims).to(device).fill_(0)
    t[:,i,:,:] = x[:,]
    return t

class ALSHVGGNet(nn.Module):
    def __init__(self, device=torch.device('cuda')):
        super(ALSHVGGNet, self).__init__()

        self.device = device
    
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1, 1)
        self.bn1   = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, 3, 1, 1, 1)
        self.bn2   = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(2)

        self.conv3 = nn.Conv2d(64, 128, 3, 1, 1, 1)
        self.bn3   = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, 3, 1, 1, 1)
        self.bn4   = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(2)

        self.conv5 = nn.Conv2d(128, 256, 3, 1, 1, 1)
        self.bn5   = nn.BatchNorm2d(256)
    
        self.conv6 = ALSHConv2d(256, 512, 3, 1, 1, 1, True, MultiHash_SRP, {},
                                4, 6, 2**4)
        
        self.bn6   = nn.BatchNorm2d(512)
        self.pool3 = nn.MaxPool2d(2)

        self.fc7   = nn.Linear(8192, 512)
        self.bn7   = nn.BatchNorm1d(512)
        self.fc8   = nn.Linear(512, 512)
        self.bn8   = nn.BatchNorm1d(512)

        self.fc9 = nn.Linear(512, 10)

    def forward(self, x):
        batch_size = x.size()[0]

        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool1(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.pool2(x)

        x = self.conv5(x)
        x = self.bn5(x)
        x = F.relu(x)
        x, i = self.conv6(x)
        x = zero_fill_missing(x, i, (batch_size, 512, 8, 8), device=self.device)

        #QU.print_mean_feature_map(x)

        x = self.bn6(x)
        x = F.relu(x)
        x = self.pool3(x)

        x = x.view(batch_size, -1) # flatten each thing in batch

        x = self.fc7(x)
        x = self.bn7(x)
        x = F.relu(x)

        x = self.fc8(x)
        x = self.bn8(x)
        x = F.relu(x)

        x = self.fc9(x)

        return x
