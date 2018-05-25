
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import pyinn as P
from alsh_conv2d_op import ALSHConv2dOp

class ALSHConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding, bias, hf, table_size, m, P, Q):
        super(ALSHConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.m = m
        self.P = P # preprocessing function
        self.Q = Q # query function

        # number of kernels * kernels_size * kernel_size * depth
        self.kernels = torch.empty(out_channels, kernel_size,
                                   kernel_size, in_channels).cuda()
        torch.nn.init.xavier_normal_(self.kernels)
        # make it a matrix where each kernel is a row.

        self.__hash = hf
        self.__table_size = table_size

        self.table = self.__build_alsh_table__()
        self.kernels = nn.Parameter(self.kernels.view(self.out_channels, -1),
                                    requires_grad=True).cuda()

        self.cache = None

    def __build_alsh_table__(self):
        # the "weight matrix" is made up of rows of flattened kernels.
        # each kernel is flattend into a row

        table = [None]*self.__table_size
        for i, kernel in enumerate(self.kernels, 0):
            kernel_flat = kernel.view(kernel.numel())
            hash_out = self.__hash(self.P(kernel_flat, self.m))
            hash_out.fmod_(self.__table_size)
            hash_out.abs_()
            index = hash_out.long().cuda()
            if table[index] is not None:
                table[index].append(i)
            else:
                table[index] = [i]

        for i in range(len(table)):
            if table[i] is None:
                table[i] = torch.Tensor([])
            table[i] = torch.Tensor(table[i]).long().cuda()

        return table

    def forward(self, x, mode):
        #if (self.cache is not None) and mode:
        #    self.rehash()
        #self.cache = x

        return ALSHConv2dOp.apply(x, self.kernels, self.kernel_size,
                                  self.stride, self.padding,
                                  self.in_channels, self.Q, self.m,
                                  self.__hash, self.table, self.__table_size)



