
import torch

class MultiSimple:
    def __init__(self, num_hashes, dim, hash_init_params):
        r"""
        which_hash can be 'StableDistribution' or 'SignRandomProjection'
        num_hashes is the number of hashes to concatenate
        hash_init_params must be a dict of {'dim':d, 'r':r}
        """

        self.bits = num_hashes
        self.a = torch.randn(dim+2, self.bits)
    

    @staticmethod
    def bits_to_int(bits):
        return (2 ** bits.nonzero()).sum()


    def P(self, x):
        app = torch.Tensor([(1 - x.norm() ** 2).sqrt(), 0])
        return torch.cat((x, app))


    def Q(self, x):
        app = torch.Tensor([0, (1 - x.norm() ** 2).sqrt()])
        return torch.cat((x, app))


    #def Q_region(self, x):
        
 

    def hash_vect(self, vect):
        r"""
        does SRP / HyperPlane LSH
        returns a torch.LongTensor
        """
        bits = torch.mm(vect.unsqueeze(0), self.a) < 0
        out = MultiSimple.bits_to_int(bits)
        return out.long()


    def query(self, input):
        return self.hash_vect(self.Q(input))


    def pre(self, input):
        return self.hash_vect(self.P(input))
