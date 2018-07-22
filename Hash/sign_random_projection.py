
import torch
import torch.nn.functional as F

import time

class SignRandomProjection():
    def __init__(self, dim, device=torch.device('cuda'), **kwargs):
        self.bits = kwargs['bits']
        self.a = torch.empty(self.bits, dim).to(device)
        self.a.normal_()
        self.device=device

    def __call__(self, x, las=None, **kwargs): 
        if x.dim() == 4:
            # if it's a batch of 3d tensors. Use conv to apply hash
            return self._dim4_hash(x, las, **kwargs)

        return self._dim2_hash(x, las)

    def _bits_to_int(self, bits):
        r"""
        Converts each of the bits of the hashes to int. 
        little-endian
        """
        i = torch.tensor(0).to(self.device)
        #mult = torch.tensor(1).to(self.device)
        for bit in bits:
            #i = (i << 1) | bit
            i = i *2 + bit
            #i += mult * bit
            #mult *= 2
        return i

        

    def _dim4_hash(self, x, las=None, **kwargs):
        r""" 
        Applies hash to a 4d object using a convoltion. 
        The convolution's paramaters are passed in kwargs
        """ 
        
        start = time.time()

        dotted = F.conv2d(x, weight=kwargs['kernel'], 
                          stride=kwargs['stride'], 
                          padding=kwargs['padding'],
                          dilation=kwargs['dilation'])
        elapsed = time.time() - start

        print('hashCONV time: ', elapsed)

        n, c, h, w = dotted.size()

        r"""
        bits represent the region x is in. 2 ** bits gives an index.
        for two points to have the same index, they must be in all the
        same regions.
        bits are little-endian. 100 is 1. 010 is 2 001 is 8, etc.
        """

        start = time.time()

        bits = (dotted > 0).long().permute(0, 2, 3, 1)

        hash = torch.Tensor(n * h * w).to(self.device)
        index = torch.tensor(0).to(self.device)

        i = torch.tensor(0).to(self.device)
        # This needs to be in a kernel
        for i in bits:
            for j in i:
                for k in j:
                    i = 0
                    for bit in k:
                        i = i*2 + bit
                    hash[index] = i
                    #hash[index] = self._bits_to_int(k)
                    '''
                    non = k.nonzero()
                    if non.sum() != 0:
                        hash[index] = (2 ** non).sum()
                    else: hash[index] = 0
                    ''' 
                    index += 1

        end = time.time() - start
        print('rest of hash ', end)

        return hash.view(n, h, w)

    def _dim2_hash(self, x, las=None):
        r""" 
        Applies hash to a vector or matrix (seq of vectors). 
        """
        dotted = self.a.mm(x)

        bits = (dotted > 0).long().transpose(0,1)

        #print(bits.size())
        hash = torch.Tensor(bits.size(0)).to(self.device)
        index = 0

        # this should be a kernel
        for row in bits:
            hash[index] = self._bits_to_int(row)
            index += 1

        return hash

