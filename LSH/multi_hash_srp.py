
import torch

class MultiHash_SRP:
    def __init__(self, num_hashes, dim, hash_init_params=None):
        r"""
        which_hash can be 'StableDistribution' or 'SignRandomProjection'
        num_hashes is the number of hashes to concatenate
        hash_init_params must be a dict of {'dim':d, 'r':r}
        """

        self.bits = num_hashes
        self.dim = dim

        # need to create vectors in first section of dim space. 
        # Then find vectors to 

        self.a = torch.randn(dim, self.bits)

        self.bit_mask = torch.Tensor([2**(i) for i in torch.arange(self.bits)])


    @staticmethod
    def bits_to_int(bits):
        return (2 ** bits.nonzero()).sum()


    def P(self, x, m=2):
        norm = x.norm()
        app = torch.Tensor([0.5 - norm**(2*(i+1)) for i in torch.arange(0,m)])
        return torch.cat((x, app))


    def P_rows(self, x):
        r'''
        assuming m = 2 simplifies the problem a lot. And it's not longer
        necessary for it to be any larger than that since Q_obj doesn't
        append 0's, it just uses a splice of self.a
        '''

        norm = x.norm(dim=1)

        norm /= norm.max() / .75

        norm.unsqueeze_(1)

        app1 = 0.5 - (norm ** 2)
        app2 = 0.5 - (norm ** 4)
        
        return torch.cat((x, app1, app2), 1)


    def hash_matr(self, matr):
        r"""
        Uses SRP on the rows a matrix.
        """
        # N x num_bits
        bits = (torch.mm(matr, self.a) > 0).float()

        return (bits * self.bit_mask).sum(1).view(-1).long()


    def hash_obj(self, obj, kernel_size, stride, padding, dilation):

        depth = obj.size(1)
        
        weight1 = self.a[:depth*kernel_size**2] # avoid appending 0
        weight1 = weight1.transpose(0, 1).view(self.bits, -1, kernel_size, kernel_size)

        out = torch.nn.functional.conv2d(obj, weight1, stride=stride, 
                                         padding=padding, dilation=dilation)

        bits = (out.view(out.size(0), -1, self.bits) > 0).float()

        hash = (bits * self.bit_mask).sum(2)

        # mode not defined for torch.cuda.tensor 
        #mode = hash.view(-1).unique().long()
        #print(mode)

        return hash


    def query(self, input, **kwargs):
        r'''
        applies Q to input and hashes. 
        If input object has dim == 4, kwargs should contaion stride,
        padding, and dilation
        '''
        assert input.dim() == 4, \
            "MultiHash_SRP.query. Input must be dim 1 or dim 4 but got: " + str(input.dim())
        return self.hash_obj(input, **kwargs)
        

    def pre(self, input):
        assert input.dim() == 2, \
            "MultiHash_SRP.pre. Input must be dim 2."
        return self.hash_matr(self.P_rows(input))


