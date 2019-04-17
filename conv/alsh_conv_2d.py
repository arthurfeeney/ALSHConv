import torch
import torch.nn as nn
from conv.alsh_conv import ALSHConv


def zero_fill_missing(x, i, dims, device):
    r"""
    fills channels that weren't computed with zeros.
    """
    t = torch.empty(dims).to(x).fill_(0)
    t[:, i, :, :] = x[:, ]
    return t


class ALSHConv2d(nn.Conv2d, ALSHConv):

    LAS = None  # static class variable to track last called layer's active set.

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding,
                 dilation,
                 bias,
                 which_hash,
                 hash_init_params,
                 K,
                 L,
                 final_L,
                 max_bits,
                 device='cpu'):
        super().__init__(in_channels,
                         out_channels,
                         kernel_size,
                         stride=stride,
                         padding=padding,
                         dilation=dilation,
                         bias=bias)

        alsh_dim = in_channels * kernel_size * kernel_size
        self.init_ALSH(L,
                       final_L,
                       max_bits,
                       which_hash,
                       hash_init_params,
                       K,
                       alsh_dim + 2,
                       out_channels,
                       device=device)
        self.cpu()
        # cache is used for modifying ALSH tables after an update.
        self.cache = None

        self.first = False
        self.last = False

    @staticmethod
    def build(conv, which_hash, hash_init_params, K, L, final_L, max_bits):
        r'''
        builds the ALSH conv from an existing convolution.
        '''
        tmp = ALSHConv2d(conv.in_channels, conv.out_channels,
                         conv.kernel_size[0], conv.stride, conv.padding,
                         conv.dilation, conv.bias is not None, which_hash,
                         hash_init_params, K, L, final_L, max_bits)
        tmp.weight.data = conv.weight.data
        return tmp

    def use_naive(self):
        '''
        when first and last are true, it fills output with zeros rather than
        sharing the last active set.
        '''
        self.first = True
        self.last = True

    def reset_freq(self):
        self.bucket_stats.reset()

    def avg_bucket_freq(self):
        return self.bucket_stats.avg

    def sum_bucket_freq(self):
        return self.bucket_stats.sum

    def cuda(self, device=None):
        r'''
        moves to specified GPU device. Also sets device used for hashes.
        '''
        for t in range(len(self.tables.hashes)):
            self.tables.hashes[t].a = self.tables.hashes[t].a.cuda(device)
            self.tables.hashes[t].bit_mask = self.tables.hashes[t].bit_mask.\
                cuda(device)
        self.device = device
        return self._apply(lambda t: t.cuda(device))

    def cpu(self):
        r'''
        moves to the CPU. Also sets device used for hashes.
        '''
        for t in range(len(self.tables.hashes)):
            self.tables.hashes[t].a = self.tables.hashes[t].a.cpu()
            self.tables.hashes[t].bit_mask = self.tables.hashes[
                t].bit_mask.cpu()
        self.device = torch.device('cpu')
        return self._apply(lambda t: t.cpu())

    def fix(self):
        self.fill_table(self.weight)

    def forward(self, x):
        r'''
        Forward pass of ALSHConv2d.
         -  x is a 4D tensor, I.e., a batch of images.
         -  LAS (optional) is the indices of the last active set.
            It specifies which kernels this conv
            should use.
        '''

        LAS = ALSHConv2d.LAS if not self.first else None

        AS, ti = self.get_active_set(x, self.kernel_size[0], self.stride,
                                     self.padding, self.dilation, LAS)

        if AS.size(0) < 2:
            AK = self.weight
        else:
            if self.first:
                # if its the first ALSHConv2d in the network,
                # then there is no LAS to use!
                AK = self.weight[AS]
            else:
                AK = self.weight[AS][:, ALSHConv2d.LAS]

        print(str(AS.size(0)) + '/' + str(self.weight.size(0)))

        output = nn.functional.conv2d(x,
                                      AK,
                                      bias=self.bias[AS],
                                      stride=self.stride,
                                      padding=self.padding,
                                      dilation=self.dilation)

        h, w = output.size()[2:]

        if self.last:
            out_dims = (x.size(0), self.out_channels, h, w)
            return zero_fill_missing(output, AS, out_dims, device=self.device)

        else:
            ALSHConv2d.LAS = AS
            return output
