
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
from Utility.cp_utils import count_votes, rehash_alsh_table, get_unique
from torch.nn.modules.utils import _pair
from Hash.scale_under_u import ScaleUnder_U
from Hash.multi_hash import MultiHash

from ALSHLayers.F_alsh_conv import ALSHConv

import time


class F_ALSHConv2d(nn.Conv2d, ALSHConv):
    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding, dilation, bias, table_size, which_hash, 
                 num_hashes, m, U, P, Q, device=torch.device('cuda'), 
                 **kwargs):
        # init conv2d. ALSHConv has no init function
        super().__init__(in_channels, out_channels, kernel_size, stride,
                         padding=padding, dilation=dilation, bias=bias)

        self.device = device

        hash_dim = kernel_size**2*in_channels + m

        self._hash = MultiHash(which_hash, num_hashes, dim=hash_dim,
                               device=self.device, **kwargs)
        self._table_size = table_size
        self.m = m
        self.P = P # preprocessing function, must take batch of images
        self.Q = Q # query function, must take batch of images
        self.U = U

        self.num_kernels = self.weight.size(0)

        self.init_table(self.weight.to(device), self._hash, Q, P, m, U,
                        table_size, out_channels, device=self.device)

        self.active_set, self.index, self.rows = None, None, None

    def _random_rows(self):
        lower, upper = 0, self.num_kernels-1
        rows = torch.LongTensor(self._table_size).random_(lower, upper)
        return get_unique(rows.sort()[0].to(self.device), self.device)

    def _generate_active_set(self, input, las):
        if las is None:
            kernels = self.weight
        else:
            kernels = self.weight[:,las]

        self.active_set, self.index, self.rows = \
            self.get_active_set(kernels, input, self._hash, self._table_size,
                                self.Q, self.P, self.m, self.in_channels,
                                self.kernel_size, self.stride, self.padding,
                                self.dilation, las, device=self.device)

    def _empty_row_case(self, las):
        # if the bucket is empty, use a random subset of kernels and
        # specify that everything should be rehashed (-1 flag).
        self.index = torch.tensor(-1).to(self.device)
        self.rows = self._random_rows()
        kernels = self.weight[:,las].to(self.device)
        self.active_set = kernels[self.rows]

    def _get_scale(self):
        #num = float(self.rows.size(0))
        #s = torch.tensor(num / self.out_channels).to(self.device)
        num = self._hash.num_hashes
        denom = self._table_size
        return torch.tensor(num / denom).to(self.device)

    def forward(self, input, las=None):
        r"""
        takes input and the last layers active set. 
        if las is None, all kernels channels are used. 
        Otherwise, only channels in las are used.
        """
        #if not self.training:
            # if testing, just run through it, don't get active set! 
        #    return F.conv2d(input, self.weight, self.bias, self.stride,
        #                    self.padding, self.dilation), None

        start = time.time()
        if self.training and self.rows is not None:
            # if training and rows exists (not first pass,) rehash
            if (self.index == -1).sum() == 0:
                # random indices were used, need to rehash everything.
                self.init_table(self.weight.to(self.device), self._hash, 
                                self.Q, self.P, self.m, self.U, 
                                self._table_size, self.out_channels, 
                                device=self.device)
            else:
                self.rehash_table(self.weight.to(self.device), self._hash, 
                                  self.P, self.m, self._table_size, 
                                  self.index, self.rows, device=self.device)
        
        rehash_time = time.time() - start

        start = time.time()
        self._generate_active_set(input, las)
        get_as_time = time.time() - start

        if self.rows.size(0) == 0:
            self._empty_row_case(las)

        start = time.time()
        output = F.conv2d(input, self.active_set, self.bias, self.stride,
                       self.padding, self.dilation)
        conv_time = time.time() - start


        print('rehash: ', rehash_time)
        print('get as: ', get_as_time)
        print('conv time: ', conv_time)
        print('total: ', rehash_time + get_as_time + conv_time)

        return output, self.rows
