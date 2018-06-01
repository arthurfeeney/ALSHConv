
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import pyinn as P
import cupy as cp
from math import sqrt

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

        self.reset_parameters()

        self._hash = hf
        self._table_size = table_size

        # make kernels a matrix where each kernel is flattened into a row.
        self.kernels = nn.Parameter(self.kernels.view(self.out_channels,
                                                      -1),
                                    requires_grad=True).cuda()

        self.table, self.table_row_lengths = self._build_alsh_table()

        self.cache = {}

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

        # done on cpu :(
        row = 0
        for kernel in self.kernels:
            hash_out = self._hash(self.P(kernel, self.m))
            hash_out.fmod_(self._table_size)
            hash_out.abs_()
            index = hash_out.long().cuda()
            table[index, indices[index]] = torch.tensor(row).long().cuda()
            indices[index] += 1
            row += 1

        return table, indices

    def _rehash(self):
        if self.cache['rows'].size() == torch.Size([0]):
            # if no rows, everything was used, so rehash it all
            self.table, self.table_row_lengths = self._build_alsh_table()
        else:
            self.table[self.cache['index']].fill_(0)
            self.table_row_lengths[self.cache['index']] = 0
            # this is done on cpu :(
            for row in self.cache['rows']:
                hash_out = self._hash(self.P(self.kernels[row], self.m))
                hash_out.fmod_(self._table_size)
                hash_out.abs_()
                index = hash_out.long().cuda()
                self.table[index, self.table_row_lengths[index]] = row
                self.table_row_lengths[index] += 1

    def _simp_forward(self, input, mode):
        if self.cache and mode:
            self._rehash()

        # dimensions of the input
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
        # viewed - permute just does transposes. transposes just
        # change indexing method so not slow.
        patches = patches.permute(1,2,3,0,4,5).contiguous().cuda()


        # reform it as a 2d matrix.
        patch_matr = patches.view(self._kernel_span, -1).cuda()

        rpc = torch.rand(5).long().cuda() * (patch_matr.size()[1]-1)
        rand_cols = patch_matr[:,rpc]
        rand_cols.transpose_(0,1)
        ave_col = rand_cols.mean(0)

        # make the column vector a unit vector
        unit_row_vect = ave_col / torch.norm(ave_col)

        # Hash the unit vector
        hash_out = self._hash(self.Q(unit_row_vect, self.m))
        hash_out.fmod_(self._table_size)
        hash_out.abs_()
        index = hash_out.long()

        # save table index for rehashing
        self.cache['index'] = index

        # get the "active set" of the kernels.
        rows = self.table[index,:self.table_row_lengths[index]]
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
        out = out.view(num_inputs, oc, h2, w2)

        #if self.bias is not None:
        #    if rows is not None:
        #        return (out + self.bias[rows].expand_as(out)), rows
        #    else:
        #        return (out + self.bias.expand_as(out)), rows
        if rows.size() == torch.Size([0]):
            return out, None
        return out, rows

    @staticmethod
    def true_las(las, kernel_size):
        r"""
        las is a 1-d tensor of active rows used by the previous layer.
        """
        start_indices = las*(kernel_size**2)
        back_indices = (las+1)*(kernel_size**2)

        new_las = torch.empty(las.size()[0]*kernel_size**2).cuda()

        for i in range(las.size()[0]):
            f = i*kernel_size**2
            b = (i+1)*kernel_size**2
            new_las[f:b] = torch.range(las[i], las[i]+kernel_size**2-1)

        return new_las.long().cuda()

    @staticmethod
    def final_las(las, h, w):
        start_indices = las*(h*w)
        back_indices = (las+1)*(h*w)

        new_las = torch.empty(las.size()[0]*h*w).cuda()

        for i in range(las.size()[0]):
            f = i*h*w
            b = (i+1)*h*w
            new_las[f:b] = torch.range(las[i], las[i]+h*w-1)

        return new_las.long().cuda()

    def _las_forward(self, input, mode, las):
        r"""
        forward pass using a subset of the channels in each
        kernel.
        """
        if self.cache and mode:
            self._rehash()

        # dimensions of the input
        input_dims = input.size()
        num_inputs = input_dims[0]
        h1 = input_dims[2]
        w1 = input_dims[3]

        # height and width of the output
        h2 = (h1 - self.kernel_size + 2*self.padding) // self.stride + 1
        w2 = (w1 - self.kernel_size + 2*self.padding) // self.stride + 1

        l = ALSHConv2d.true_las(las, self.kernel_size)

        patches = P.im2col(input, self.kernel_size, self.stride,
                           self.padding)

        # change ordering so columns correspond to kernel regions when
        # viewed - permute just does transposes. transposes just
        # change indexing method so not slow.
        patches = patches.permute(1,2,3,0,4,5).contiguous().cuda()

        # reshape patches as a matrix
        patch_matr = patches.view(l.size()[0], -1)

        rpc = torch.rand(5).long().cuda() * (patch_matr.size()[1]-1)
        rand_cols = patch_matr[:,rpc]
        rand_cols.transpose_(0,1)
        ave_col = rand_cols.mean(0)

        # make the column vector a unit vector
        unit_row_vect = ave_col / torch.norm(ave_col)


        urv_len = unit_row_vect.size()[0]
        m_indices = torch.range(urv_len, urv_len+self.m-1).long().cuda()
        x_non_zero = torch.cat((l, m_indices)).cuda()


        # Hash the unit vector
        hash_out = self._hash(self.Q(unit_row_vect, self.m), x_non_zero)
        hash_out.fmod_(self._table_size)
        hash_out.abs_()
        index = hash_out.long()

        # save table index for rehashing
        self.cache['index'] = index

        # get the "active set" of the kernels.
        rows = self.table[index,:self.table_row_lengths[index]]
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
            oc = active[rows].size()[0]

            out = sub_out
            scale = rows.size()[0] / self.kernels.size()[0]
            out /= scale # need to scale values in output b/c there's less
        else:
            # if there's no rows in the bucket, just use entire kernel.
            out = active.mm(patch_matr).cuda()
            oc = active.size()[0]

        scale2 = l.size()[0] / self.kernels.size()[1]

        out /= scale2

        # O x N x (h2*w2)
        out = out.view(oc, num_inputs, h2*w2)

        # N x O x (h2*w2)
        out.transpose_(0,1)

        # N x O x h2 x w2 - proper output dims
        out = out.view(num_inputs, oc, h2, w2)

        #if self.bias is not None:
        #    if rows is not None:
        #        return (out + self.bias[rows].expand_as(out)), rows
        #    else:
        #        return (out + self.bias.expand_as(out)), rows
        if rows.size() == torch.Size([0]):
            return out, None
        return out, rows

    def forward(self, input, mode, las=None):
        if las is not None:
            return self._las_forward(input, mode, las)
        else:
            return self._simp_forward(input, mode)
