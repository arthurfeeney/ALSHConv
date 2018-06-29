

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.Parameter as Parameter

class F_BatchNorm2d(nn.Module):
    def __init__(self, num_features, affine=True):
        super(F_BatchNorm2d, self).__init__()

        self.affine = affine



        self.weight = Parameter(torch.Tensor(num_features))
        self.bias = Parameter(torch.Tensor(num_featres))

        self.reset_parameters()

    def reset_parameters(self):
        self.running_mean.zero()
        self.running_var.fill_(1)

        self.weight.data.uniform_()
        self.bias.data.zero_()

    def forward(self, input, las):
        return F.batch_norm(input, self.running_mean[las], self.running_var[las])
