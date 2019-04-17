
import torch
from hash_utils import tensor_to_int

class MultiHash_SD:
    def __init__(self, num_hashes, dim, hash_init_params):
        r"""
        which_hash can be 'StableDistribution' or 'SignRandomProjection'
        num_hashes is the number of hashes to concatenate
        hash_init_params must be a dict of {'dim':d, 'r':r}
        """

        assert isinstance(hash_init_params, dict), \
                'hash_init_params must be a dict.'
        assert (len(hash_init_params) == 1 and 
                'r' in hash_init_params)

        with hash_init_params['r'] as r:
            self.A = torch.randn(dim, num_hashes)
            self.R = torch.tensor(r)
            self.B = torch.rand(num_hashes * self.R)
            self.num_hashes = num_hashes
    
    def hash_1d(self, input):
        r"""
        hashes a vector
        """
        prod = torch.mm(input.unsqueeze(0), self.A)
        sum = prod + self.B
        div = torch.floor(sum / self.R)
        div.abs_()

        out = tensor_to_int(div, device=self.device)

        return out.long()

    def hash_rows(self, input):
        r"""
        hashes each rows of a matrix.
        """
        prod = torch.mm(input, self.A)
        sum = prod + self.B.expand_as(prod)

        # [input.size(0)] x [num_hashes]
        div = torch.floor(sum / self.R)
        div.abs_()

        out = tensor_to_int(div, device=self.device)

        return out.long()

    def __call__(self, input):
        if input.dim() == 1:
            return self.hash_1d(input)
        if input.dim() == 2:
            return self.hash_rows(input)




