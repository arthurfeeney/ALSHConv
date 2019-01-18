
import torch
import torch.nn as nn
from Conv.alsh_conv import ALSHConv

def zero_fill_missing(x, i, dims, device):
    r"""
    fills channels that weren't computed with zeros.
    """
    t = torch.empty(dims).to(x).fill_(0)
    t[:,i,:,:] = x[:,]
    return t

class ALSHConv2d(nn.Conv2d, ALSHConv):
    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding, dilation, bias, which_hash, hash_init_params,
                 K, L, max_bits, device='cpu'):
        super().__init__(in_channels, out_channels, kernel_size,
                         stride=stride, padding=padding, dilation=dilation,
                         bias=bias)

        alsh_dim = in_channels * kernel_size * kernel_size
        self.init_ALSH(L, max_bits, which_hash, hash_init_params, K,
                       alsh_dim+2, out_channels, device=device)
        self.cpu()
        # cache is used for modifying ALSH tables after an update.
        self.cache = None


    @staticmethod
    def build(conv, which_hash, hash_init_params, K, L, max_bits):
        r'''
        builds the ALSH conv from an existing convolution.
        '''
        tmp = ALSHConv2d(conv.in_channels, conv.out_channels, conv.kernel_size[0],
                         conv.stride, conv.padding, conv.dilation,
                         conv.bias is not None, which_hash, hash_init_params,
                         K, L, max_bits)
        tmp.weight.data = conv.weight.data
        return tmp

    def cuda(self, device=None):
        r'''
        moves to specified GPU device. Also sets device used for hashes.
        '''
        print('using device gpu!')
        for t in range(len(self.tables.hashes)):
            self.tables.hashes[t].a = self.tables.hashes[t].a.cuda(device)
            self.tables.hashes[t].bit_mask = self.tables.hashes[t].bit_mask.\
                cuda(device)
        self.device = device
        return self._apply(lambda t: t.cuda(device))

    def cpu(self):
        for t in range(len(self.tables.hashes)):
            self.tables.hashes[t].a = self.tables.hashes[t].a.cpu()
            self.tables.hashes[t].bit_mask = self.tables.hashes[t].bit_mask.cpu()
        self.device = torch.device('cpu')
        return self._apply(lambda t: t.cpu())

    def ALSH_mode(self):
        self.fill_table(self.weight)

    def fix(self):
        self.fill_table(self.weight)

    def forward(self, x, LAS=None):
        r'''
        Forward pass of ALSHConv2d.
         -  x is a 4D tensor, I.e., a batch of images.
         -  LAS (optional) is the indices of the last active set.
            It specifies which kernels this conv
            should use.
        '''

        #if self.cache is not None and self.training:
        #    active_kernels, active_set, table_indices = self.cache
        #    self.refill(active_kernels, active_set, table_indices)

        AS, ti = self.get_active_set(x, self.kernel_size[0], self.stride,
                                     self.padding, self.dilation)

        #print(str(AS.size(0)) + ' of ' + str(self.weight.size(0)))

        if LAS is None:
            AK = self.weight[AS]
        else:
            AK = self.weight[AS][:,LAS] # weight[AS, LAS] doesnt seem to work...

        output = nn.functional.conv2d(x, AK, self.bias[AS],
                                      self.stride, self.padding,
                                      self.dilation)

        #self.cache = AK, AS, ti

        h, w = output.size()[2:]

        # scale by the inverse of the fraction of filters used.
        scale = AS.size(0) / self.out_channels
        output /= scale

        out_dims = (x.size(0), self.out_channels, h, w)

        return zero_fill_missing(output, AS, out_dims, device=self.device) #, AS
