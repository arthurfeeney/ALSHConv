
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import pyinn as P
from alsh_op import ALSHOp

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
        torch.nn.init.xavier_normal_(self.kernels)

        self.__hash = hf
        self.__table_size = table_size

        self.table = self.__build_alsh_table__()

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

        input_dims = x.size()
        num_inputs = input_dims[0]
        h1 = input_dims[1]
        w1 = input_dims[2]


        # height and width of the output
        h2 = (h1 - self.kernel_size + 2*self.padding) / self.stride + 1
        w2 = (w1 - self.kernel_size + 2*self.padding) / self.stride + 1

        kernel_elem = self.kernel_size**2 * self.in_channels

        # make input patches.
        y = P.im2col(x.cuda(), self.kernel_size, self.stride, self.padding)
        y = y.permute(0,1,4,2,3,5)
        #y = y.view(y.size()[0:-1])

        #y = y.view(self.out_channels, -1).cuda()

        # make kernels a row matrix.
        weight = self.kernels.view(self.out_channels, -1).cuda()

        #y = ALSHOp.apply(y, weight, self.Q, self.m, self.__hash, self.table,
        #                 self.__table_size, True).cuda()

        trans = torch.Tensor(kernel_elem, num_inputs * h2 * w2)
        trans = trans.cuda()

        print(y.size())

        # make input a column matrix
        c = 0
        for image in y:
            for col in image:
                for d in range(kernel_elem / 3):
                #for t in col:
                    t = col[d*3:((d+1)*3)]
                    input_col = t.contiguous().view(-1)
                    trans[:, c] = input_col
                    c += 1

        print(weight.size(), trans.size())

        # compute output
        out = weight.mm(trans)

        # reshape output into an image
        out = out.view(num_inputs, h2, w2, self.out_channels)


        # reshape output column matrix into image
        #out = P.col2im(out, self.kernel_size, self.stride, self.padding)

        return out


