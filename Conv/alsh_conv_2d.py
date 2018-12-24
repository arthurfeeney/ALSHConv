
import torch
import torch.nn as nn
from Conv.alsh_conv import ALSHConv

def zero_fill_missing(x, i, dims, device=torch.device('cuda')):
    r"""
    fills channels that weren't computed with zeros.
    """
    if i is None:
        return x
    t = torch.empty(dims).to(device).fill_(0)
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

        #self.fill_table(self.weight)

        self.cache = None

        self.device = device


    def ALSH_mode(self):
        self.fill_table(self.weight)


    def forward(self, x, LAS=None):
        r'''
        trains regularly, prunes during inference. 
        '''
        if not self.training:
            return self.forward2(x, LAS)
        else:
            return (nn.functional.conv2d(x, self.weight, self.bias,
                                        self.stride, self.padding, 
                                        self.dilation), None)

    
    def forward2(self, x, LAS):
        #if self.cache is not None and self.training:
        #    active_kernels, active_set, table_indices = self.cache
        #    self.refill(active_kernels, active_set, table_indices)

        AS, ti = self.get_active_set(x, self.kernel_size[0], self.stride, 
                                     self.padding, self.dilation) 

        #AK = self.weight[AS]
        if LAS is None:
            AK = self.weight[AS]
        else:
            AK = self.weight[AS][:,LAS]

        #print(AK.size())

        output = nn.functional.conv2d(x, 
                                      AK, self.bias[AS], 
                                      self.stride, self.padding, 
                                      self.dilation)

        #self.cache = AK, AS, ti

        #output[output < 0.05] = 0

        h, w = output.size()[2:]

        out_dims = (x.size(0), self.out_channels, h, w)

        return zero_fill_missing(output, AS, out_dims, self.device)  #, AS
