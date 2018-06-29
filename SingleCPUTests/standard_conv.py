
import torch
import torch.nn as nn
import torch.nn.functional as F
import ALSHLayers.F_alsh_conv2d as F_Conv2d

class ManyFilterNet(nn.Module):
    def __init__(self):
        super(ManyFilterNet, self).__init__()

        self.conv1 = nn.Conv2d(3, 1000, 3, 1, 1, 1, bias=False)
        self.conv2 = nn.Conv2d(1000, 1000, 3, 1, 1, 1, bias=False)

        self.fc1 = nn.Linear(1024000, 10)


    def forward(self, x):
        batch_size = x.size()[0]
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(batch_size, -1)
        return self.fc1(x)
