
import torch

class MultiHash:
    def __init__(self, which_hash, num_hashes, **kwargs):
        self.hashes = [which_hash(**kwargs) for _ in range(num_hashes)]
        self.num_hashes = num_hashes
        self.device = kwargs['device']

        if 'bits' in kwargs:
            self.bits = kwargs['bits']
        else: 
            self.bits = 1

    r"""
    a bunch of this stuff probably needs to be CUDA kernels eventually...
    """
    def __call__(self, x, rows=None, **kwargs):
        if kwargs:
            kernels = kwargs['kernel']
            stride = kwargs['stride']
            padding = kwargs['padding']
            dilation = kwargs['dilation']
            out = torch.stack([self.hashes[h](x, rows, kernel=kernels[h], 
                                              stride=stride, 
                                              padding=padding, 
                                              dilation=dilation) \
                             for h in range(self.num_hashes)]).to(self.device)
        else:
            out = torch.stack([h(x, rows) for \
                              h in self.hashes]).to(self.device)
        return out

    def all_b(self):
        return torch.Tensor([h.b for h in self.hashes]).to(self.device)

    def a_to_kernel(self, d, kernel_size, m):
        r"""
        Converts all of the hashes a into filters that can be used in a
        convolution.
        """   
        if self.bits > 1:
            flat = torch.stack([h.a[:,:-m] for h in self.hashes])
            flat = flat.to(self.device)
        else:
            flat = torch.stack([h.a for h in self.hashes]).to(self.device)
        print(flat)
        return flat.view(self.num_hashes,self.bits, 
                         d, kernel_size[0], kernel_size[1])

