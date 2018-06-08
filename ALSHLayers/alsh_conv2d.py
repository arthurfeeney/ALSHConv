
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import pyinn as P
import cupy as cp
from math import sqrt
from cp_utils import count_votes, get_true_las

class ALSHConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding, bias, hf, table_size, m, P, Q):
        super(ALSHConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self._kernel_span = kernel_size**2 * in_channels

        self.m = m
        self.P = P # preprocessing function
        self.Q = Q # query function

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

        self._hash = hf
        self._table_size = table_size

        # make kernels a matrix where each kernel is flattened into a row.
        self.kernels = nn.Parameter(self.kernels.view(self.out_channels,
                                                      -1),
                                    requires_grad=True).cuda()

        self.table, self.table_row_lengths = self._build_alsh_table()

        self.cache = {}

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

    def _build_alsh_table(self):
        table = torch.empty(self._table_size,
                            self._kernel_span).long().cuda().fill_(0)

        indices = torch.empty(self._table_size).long().cuda().fill_(0)

        a_row_matr = self._hash.a.view(1, self._hash.a.size()[0])
        idx = a_row_matr.mm(self.P(self.kernels, self.m).transpose(0,1))
        idx = idx.view(-1).cuda()
        idx = torch.floor((idx + self._hash.b) / self._hash.r)
        idx = idx.long()

        idx.fmod_(self._table_size)
        idx.abs_()

        for i, index in enumerate(idx):
            table[index, indices[index]] = i
            indices[index] += 1

        return table, indices

    def _rehash(self):
        if self.cache['rows'].size() == torch.Size([0]):
            # if no rows, everything was used, so rehash it all
            self.table, self.table_row_lengths = self._build_alsh_table()
        else:
            self.table_row_lengths[self.cache['index']] = 0
            rows = self.cache['rows']
            # this is done on cpu :(
            a_row_matr = self._hash.a.view(1, self._hash.a.size()[0])
            indices = a_row_matr.mm(self.P(
                                            self.kernels[rows], self.m
                                          ).transpose(0,1)
                                   )
            indices = indices.view(-1).cuda()
            indices = torch.floor((indices + self._hash.b) / self._hash.r)
            indices = indices.long()

            indices.fmod_(self._table_size)
            indices.abs_()

            # there aren't that many indices, so this should be fine
            for i, index in enumerate(indices):
                self.table[index, self.table_row_lengths[index]] = rows[i]
                self.table_row_lengths[index] += 1


    def _vote(self, input, las=None):
        r"""
        function hashes all columns of input and returns the most common
        bucket.
        """

        if las is None:
            # called from _simp_forward
            a_row_matr = self._hash.a.view(1, self._hash.a.size()[0])
            votes = a_row_matr.mm(self.Q(input, self.m))
            votes = torch.floor((votes + self._hash.b) / self._hash.r)
            votes = votes.view(-1).long().cuda()

            counts = count_votes(votes, self._table_size)

            return torch.argmax(counts)
        else:
            a_size = self._hash.a.size()[0]
            m_indices = torch.range(a_size-self.m, a_size-1).long().cuda()

            las = torch.cat((las, m_indices)).long().cuda()

            a_row_matr = self._hash.a[las].view(1, las.size()[0])
            votes = a_row_matr.mm(self.Q(input, self.m))
            votes = torch.floor((votes + self._hash.b) / self._hash.r)
            votes = votes.view(-1).long().cuda()

            counts = count_votes(votes, self._table_size)

            return torch.argmax(counts)


    def _simp_forward(self, input, mode):
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
        patch_matr = patches.contiguous().view(self._kernel_span, -1).cuda()

        index = self._vote(patch_matr).long().cuda()

        # save table index for rehashing
        self.cache['index'] = index

        # get the "active set" of the kernels.
        rows = self.table[index,0:self.table_row_lengths[index]]
        rows = rows.view(-1)

        # save rows for rehashing
        self.cache['rows'] = rows

        oc = 0
        out = torch.Tensor([]).cuda()
        if rows.size() != torch.Size([0]):
            # if there are some rows in the bucket, only perform the forward
            # pass with them
            sub_out = self.kernels[rows].mm(patch_matr)
            oc = self.kernels[rows].size()[0]

            out = sub_out
            scale = rows.size()[0] / self.kernels.size()[0]
            out /= scale # need to scale values in output b/c there's less
        else:
            # if there's no rows in the bucket, just use entire kernel.
            out = self.kernels.mm(patch_matr).cuda()
            oc = self.kernels.size()[0]

        # O x N x (h2*w2)
        out = out.view(oc, num_inputs, h2*w2)

        # N x O x (h2*w2)
        out.transpose_(0,1)

        # N x O x h2 x w2 - proper output dims
        #out = out.view(num_inputs, oc, h2, w2)

        #if self.bias is not None:
        #    if rows is not None:
        #        return (out + self.bias[:,rows].expand_as(out)), rows
        #    else:
        #        return (out + self.bias.expand_as(out)), rows
        if rows.size() == torch.Size([0]):
            return out.view(num_inputs, oc, h2, w2), None
        return out.view(num_inputs, oc, h2, w2), rows

    @staticmethod
    def true_las(las, kernel_size):
        r"""
        uses CUDA kernel to get indices into matrix that correspnd to las
        """
        return get_true_las(las, kernel_size)

    def _las_forward(self, input, mode, las):
        r"""
        forward pass using a subset of the channels in each
        kernel.
        """
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

        l = ALSHConv2d.true_las(las, self.kernel_size)

        # apply im2col to the input.
        # patches is N x C x K x K X H x W
        patches = P.im2col(input, self.kernel_size, self.stride,
                           self.padding)

        # change ordering so columns correspond to kernel regions when
        # viewed - transposes just change indexing method so not slow.
        # doing this in-place to be sure there is not unnecesarry copying.
        # C x K x K x N x H x W
        patches.transpose_(0,1)
        patches.transpose_(1,2)
        patches.transpose_(2,3)

        # reshape patches as a matrix
        # patches_matr is [C x K x K] x [N x H x W]
        patch_matr = patches.contiguous().view(l.size()[0], -1)

        index = self._vote(patch_matr, l).long().cuda()

        # save table index for rehashing
        self.cache['index'] = index

        # get the "active set" of the kernels.
        rows = self.table[index,0:self.table_row_lengths[index]]
        rows = rows.view(-1)

        # save rows for rehashing
        self.cache['rows'] = rows

        active = self.kernels[:,l]

        oc = 0
        out = torch.Tensor([]).cuda()
        if rows.size() != torch.Size([0]):
            # if there are some rows in the bucket, only perform the forward
            # pass with them
            sub_out = active[rows].mm(patch_matr)

            oc = rows.size()[0]

            out = sub_out
            scale = rows.size()[0] / self.kernels.size()[0]
            out /= scale # scale up by number of rows useds
        else:
            # if there's no rows in the bucket, just use entire kernel.
            out = active.mm(patch_matr).cuda()
            oc = active.size()[0]

        # O x N x (h2*w2)
        out = out.view(oc, num_inputs, h2*w2)

        # N x O x (h2*w2)
        out.transpose_(0,1)

        # N x O x h2 x w2 - proper output dims
        #out = out.view(num_inputs, oc, h2, w2)

        #if self.bias is not None:
        #    if rows is not None:
        #        return (out + self.bias[:,rows].expand_as(out)), rows
        #    else:
        #        return (out + self.bias.expand_as(out)), rows
        if rows.size() == torch.Size([0]):
            return out.view(num_inputs, oc, h2, w2), None
        return out.view(num_inputs, oc, h2, w2), rows

    def forward(self, input, mode, las=None):
        if self.cache and mode:
            self._rehash()

        if las is not None:
            return self._las_forward(input, mode, las)
        else:
            return self._simp_forward(input, mode)
