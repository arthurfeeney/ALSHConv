
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import pyinn as P
from math import sqrt
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

        self.table = self._build_alsh_table()

        # make kernels a matrix where each kernel is flattened into a row.
        self.kernels = nn.Parameter(self.kernels.view(self.out_channels,
                                                      -1),
                                    requires_grad=True).cuda()


        self.cache = None

    def reset_parameters(self):
        # default init used by nn.Conv2d
        n = self.in_channels * (self.kernel_size**2)
        stdv = 1. / sqrt(n)
        self.kernels.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def _build_alsh_table(self):
        table = [None]*self._table_size

        for i, kernel in enumerate(self.kernels, 0):
            kernel_flat = kernel.view(kernel.numel())
            hash_out = self._hash(self.P(kernel_flat, self.m))
            hash_out.fmod_(self._table_size)
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

        tmp = nn.ParameterList().cuda()
        for i in table:
            tmp.append(nn.Parameter(i, requires_grad=False))

        table = tmp

        return table

    #def rehash(self):

    def get_rows(self, input):
        hash_out = self._hash(self.Q(input, m))
        hash_out.fmod_(self._table_size)
        hash_out.abs_()
        index = hash_out.long()


    def forward(self, x, mode):
        #if (self.cache is not None) and mode:
        #    self.rehash()
        #self.cache = x

        # x should be a patch matrix
        # dimensions of the input
        input_dims = x.size()
        num_inputs = input_dims[0]
        h1 = input_dims[2]
        w1 = input_dims[3]

        # height and width of the output
        h2 = (h1 - self.kernel_size + 2*self.padding) // self.stride + 1
        w2 = (w1 - self.kernel_size + 2*self.padding) // self.stride + 1

        patches = P.im2col(x, self.kernel_size, self.stride, self.padding)

        # change ordering so columns correspond to kernel regions when
        # viewed - permute just does transposes. transposes just
        # change indexing method so not slow.
        patches = patches.permute(1,2,3,0,4,5).contiguous().cuda()

        # reform it as a 2d matrix.
        patch_matr = patches.view(self.kernel_size**2 * self.in_channels,
                                  -1).cuda()

        ave_im_patch = torch.zeros(patch_matr[:,:h2*w2].size())

        # transposing inplace twice faster than returning copy
        patch_matr.transpose_(0,1)
        max_each_row_vect = patch_matr.max(0)[0]
        patch_matr.transpose_(0,1)

        # make the column vector a unit vector
        unit_row_vect = max_each_row_vect / torch.norm(max_each_row_vect)

        # Hash the unit vector
        hash_out = self._hash(self.Q(unit_row_vect, self.m))
        hash_out.fmod_(self._table_size)
        hash_out.abs_()
        index = hash_out.long()

        # get the "active set" of the kernels.
        rows = self.table[index]

        out = torch.Tensor([]).cuda()

        if rows.size() != torch.Size([0]):
            # if there are some rows in the bucket, only perform the forward
            # pass with them
            out = torch.empty(self.kernels.size()[0],
                              patch_matr.size()[1]).cuda().fill_(0)

            sub_out = self.kernels[rows].mm(patch_matr)

            out[rows] = sub_out
            scale = rows.size()[0] / self.kernels.size()[0]
            out /= scale # need to scale values in output b/c there's less
        else:
            # if there's no rows in the bucket, just use all of them.
            out = self.kernels.mm(patch_matr).cuda()

        # O x N x (h2*w2)
        out = out.view(self.kernels.size()[0], num_inputs, h2*w2)

        oc = out.size()[0]

        # N x O x (h2*w2)
        out.transpose_(0,1)
        #out = out.permute(1, 0, 2).contiguous()

        # N x O x h2 x w2 - proper output dims
        out = out.view(num_inputs, oc, h2, w2)

        if self.bias is not None:
            return out + self.bias.expand_as(out)
        else:
            return out


