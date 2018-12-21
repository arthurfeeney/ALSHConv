

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

class F_BatchNorm2d(nn.BatchNorm2d):
    def __init__(self, num_features, affine=True):
        super(F_BatchNorm2d, self).__init__(num_features, affine=affine)

        #self.affine = affine

        #self.weight = Parameter(torch.Tensor(num_features))
        #self.bias = Parameter(torch.Tensor(num_featres))

        #self.reset_parameters()

    #def reset_parameters(self):
    #    self.running_mean.zero()
    #    self.running_var.fill_(1)

    #    self.weight.data.uniform_()
    #    self.bias.data.zero_()

    def forward(self, input, LAS):
        if LAS is None:
            return F.batch_norm(input, self.running_mean, self.running_var)
        else:
            return F.batch_norm(input, self.running_mean[LAS], self.running_var[LAS])
