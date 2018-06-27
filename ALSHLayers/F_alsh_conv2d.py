
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import pyinn as P
from Utility.cp_utils import count_votes, rehash_alsh_table
from torch.nn.modules.utils import _pair

class ALSHConv(nn.Module):
    def init_table(self, kernels, hash, P, m, table_size, out_channels,
                   device=torch.device('cuda')):
        self.table = torch.Tensor(table_size,
                                  kernels.size()[0]*2).to(device).long().fill_(0)
        self.table_row_lengths = \
            torch.Tensor(table_size).to(device).long().fill_(0)

        indices = \
            hash(P(kernels, m, device).transpose(0,1),).view(-1).long().to(device)
        indices.fmod_(table_size)
        indices.abs_()

        i = 0
        for idx in indices:
            self.table[idx, self.table_row_lengths[idx]] = i
            self.table_row_lengths[idx] += 1
            i += 1

    def rehash_table(self, kernels, hash, P, m, table_size, index, rows,
                     device=torch.device('cuda')):
        self.table_row_lengths[index] = 0

        kernels = kernels[rows] # updated weights in last active set

        indices = hash(P(kernels, m, device).transpose(0,1), rows).view(-1).long()
        indices.fmod_(table_size)
        indices.abs_()

        i = 0
        for idx in indices:
            self.table[idx, self.table_row_lengths[idx]] = rows[i]
            self.table_row_lengths[idx] += 1
            i += 1

    def get_active_set(self, kernels, input, hash, table_size, Q, m,
                       in_channels, kernel_size, stride, padding, dilation,
                       las=None, device=torch.device('cuda')):
        votes = self.vote(input, hash, Q, m, in_channels, kernel_size,
                          stride, padding, dilation, las, device=device)

        votes.fmod_(table_size)
        votes.abs_()

        if device == torch.device('cuda'):
            count = count_votes(votes, table_size)
        else:
            count = torch.Tensor(table_size)
            for v in votes:
                count[v.long()] += 1

        index = count.argmax()

        rows = self.table[index, 0:self.table_row_lengths[index]]


        if rows.size() != torch.Size([0]):
            rows = rows.sort()[0]

        active_set = kernels[rows]
        return active_set, index, rows


    def vote(self, input, hash, Q, m, in_channels, kernel_size, stride,
             padding, dilation, las=None, device=torch.device('cuda')):
        r"""
        instead of reshaping input using im2col it, applies the hashes dot
        product to the input using a convolution. Q needs to append an 'image', of
        whatever it appends, to every 'image' in the input -> this requires m to
        be a certain size. Also need to prevent hash.a from being updated!
        """

        d = in_channels + (self.m / (kernel_size[0] * kernel_size[1]))

        kernel = hash.a.view(1, d, kernel_size[0], kernel_size[1])


        if las is not None:
            las_and_last = \
                torch.cat((las, torch.range(in_channels, d-1).long().to(device)))
            kernel = kernel[:,las_and_last]

        input_Q = Q(input, m, kernel_size, device=device)


        dotted = F.conv2d(input_Q, kernel, stride=stride,
                          padding=padding, dilation=dilation)

        votes = torch.floor((dotted.view(-1) + hash.b) / hash.r)

        return votes


class F_ALSHConv2d(nn.Conv2d, ALSHConv):
    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding, dilation, bias, hf, table_size, m, P, Q,
                 device=torch.device('cuda')):
        super().__init__(in_channels, out_channels, kernel_size, stride,
                         padding=padding, dilation=dilation, bias=bias)
        self._hash = hf #, must take batch of images and return integer
        self._table_size = table_size
        self.m = m
        self.P = P # preprocessing function, must take batch of images
        self.Q = Q # query function, must take batch of images

        self.device = device

        self.num_kernels = self.weight.size()[0]

        self.init_table(self.weight.to(device), hf, P, m, table_size,
                        out_channels, device=self.device)

        self.active_set, self.index, self.rows = None, None, None

    def random_rows(self):
        rows = torch.LongTensor(self._table_size).random_(0, self.num_kernels-1)
        return rows.to(self.device)

    def forward(self, input, las=None):
        if self.training and self.rows is not None:
            self.rehash_table(self.weight.to(self.device), self._hash, self.P,
                              self.m, self._table_size, self.index, self.rows,
                              device=self.device)
            #self.init_table(self.weight.to(self.device), self._hash, self.P,
            #                self.m, self._table_size, self.out_channels,
            #                device=self.device)

        if las is None:
            kernels = self.weight
        else:
            kernels = self.weight[:,las]


        self.active_set, self.index, self.rows = \
            self.get_active_set(kernels, input, self._hash, self._table_size,
                                self.Q, self.m, self.in_channels,
                                self.kernel_size, self.stride, self.padding,
                                self.dilation, las, device=self.device)

        if self.rows.size() == torch.Size([0]):
            self.rows = self.random_rows()
            self.active_set = kernels[self.rows]

        out = F.conv2d(input, self.active_set, self.bias, self.stride,
                       self.padding, self.dilation)


        # it may not be necessary to scale if still dropping at test time?
        scale = torch.tensor(float(self.rows.size()[0]) /
                             self.out_channels).to(self.device)
        out /= scale

        return out, self.rows

