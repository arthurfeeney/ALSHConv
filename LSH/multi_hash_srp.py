
import torch
import torch.nn as nn
import numpy

class MultiHash_SRP(nn.Module):
    def __init__(self, num_hashes, dim, hash_init_params=None):
        r"""
        which_hash can be 'StableDistribution' or 'SignRandomProjection'
        num_hashes is the number of hashes to concatenate
        """
        self.bits = num_hashes
        self.dim = dim

        self.a = torch.randn(dim, self.bits)

        self.normal = self.a[:-2]
        self.bit_mask = torch.Tensor([2**(i) for i in torch.arange(self.bits)])


    def P(self, x, m=2):
        norm = x.norm()
        app = torch.Tensor([0.5 - norm**(2*(i+1)) for i in torch.arange(0,m)])
        return torch.cat((x, app))


    def P_rows(self, x):
        r'''
        assuming m = 2 simplifies the problem a lot. And it's not longer
        necessary for it to be any larger than that since Q_obj doesn't
        append 0's, it just uses a splice of self.a.

        - x should be a matrix where rows are the datum to be inserted.
        '''

        norm = x.norm(dim=1) # norm of each row

        norm /= norm.max() / .75 # scale magnitude of each row < .75

        norm.unsqueeze_(1)

        app1 = 0.5 - (norm ** 2)
        app2 = 0.5 - (norm ** 4)

        return torch.cat((x, app1, app2), 1)


    def hash_matr(self, matr):
        r"""
        Applies SRP hash to the rows a matrix.
        """
        # N x num_bits
        bits = (torch.mm(matr, self.a.to(matr)) > 0).float()
        return (bits * self.bit_mask.to(matr)).sum(1).view(-1).long()


    def hash_4d_tensor(self, obj, kernel_size, stride, padding, dilation,
                       LAS=None):

        # having normal=a[:-2] instead of a prevents
        # some copying each batches
        normal = self.normal.transpose(0, 1).view(self.bits, -1, kernel_size,
                                                  kernel_size).to(obj)

        if LAS is not None:
            normal = normal[:,LAS]

        out = torch.nn.functional.conv2d(obj, normal, stride=stride,
                                         padding=padding, dilation=dilation)

        trs = out.view(out.size(0), self.bits, -1).transpose(1,2)
        bits = (trs >= 0).float()
        return (bits * self.bit_mask.to(obj)).sum(2)

    def query(self, input, **kwargs):
        r'''
        applies Q to input and hashes.
        If input object has dim == 4, kwargs should contaion stride,
        padding, and dilation
        '''
        assert input.dim() == 4, \
            "MultiHash_SRP.query. Input must be dim 4 but got: " +\
            str(input.dim())
        return self.hash_4d_tensor(input, **kwargs)

    def pre(self, input):
        assert input.dim() == 2, \
            "MultiHash_SRP.pre. Input must be dim 2."
        return self.hash_matr(self.P_rows(input))


