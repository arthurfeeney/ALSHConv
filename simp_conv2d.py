
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import pyinn as P
import cupy as cp
from math import sqrt

class SimpConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding, bias):
        super(SimpConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self._kernel_span = kernel_size**2 * in_channels

        # number of kernels * kernels_size * kernel_size * depth
        self.kernels = nn.Parameter(torch.empty(out_channels, kernel_size,
                                    kernel_size, in_channels),
                                    requires_grad=True).cuda()

        if bias:
            self.bias = nn.Parameter(torch.zeros(1, out_channels, 1, 1),
                                     requires_grad=True).cuda()
        else:
            self.bias = None

        #self.reset_parameters()
        self.init_xavier_uniform()


        # make kernels a matrix where each kernel is flattened into a row.
        self.kernels = nn.Parameter(self.kernels.view(self.out_channels,
                                                      -1),
                                    requires_grad=True).cuda()

    def init_xavier_uniform(self):
        nn.init.xavier_uniform_(self.kernels)
        if self.bias is not None:
            nn.init.xavier_uniform_(self.bias)

    def reset_parameters(self):
        # default init used by nn.Conv2d
        n = self.in_channels * (self.kernel_size**2)
        stdv = 1. / sqrt(n)
        self.kernels.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        # dimensions of the input
        if not input.is_contiguous():
            input = input.contiguous()

        input_dims = input.size()
        num_inputs = input_dims[0]
        h1 = input_dims[2]
        w1 = input_dims[3]

        # height and width of the output
        h2 = (h1 - self.kernel_size + 2*self.padding) // self.stride + 1
        w2 = (w1 - self.kernel_size + 2*self.padding) // self.stride + 1


        k = self.kernel_size**2

        patches = P.im2col(input, self.kernel_size, self.stride,
                           self.padding)

        # change ordering so columns correspond to kernel regions when
        # viewed - transposes just change indexing method so not slow.
        # doing this in-place so there is not unnecesarry copying.
        # C x K x K x N x H x W
        patches.transpose_(0,1)
        patches.transpose_(1,2)
        patches.transpose_(2,3)

        # reform it as a 2d matrix.
        patch_matr = patches.contiguous().view(self._kernel_span, -1)

        out = self.kernels.mm(patch_matr)

        # O x N x (h2*w2)
        out = out.view(self.out_channels, num_inputs, h2*w2)

        # N x O x (h2*w2)
        out.transpose_(0,1)

        return out.view(num_inputs, self.out_channels, h2, w2)
