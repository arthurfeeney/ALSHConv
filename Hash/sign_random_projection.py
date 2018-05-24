
import torch



# should compute H(P(T(X)))
# and H(Q(T(X))), T makes it so query doesn't need to be a unit vector.

class SignRandomProjectionHash():
    def __init__(self, dim):
        self.a = torch.empty(dim)
        torch.nn.init.normal_(self.a)

    def __call__(self, x):
        return (self.a.dot(x) > 0).long()


def append_half_minus_norm(x, m):
    # preprocessing function
    x_n = torch.norm(x)
    powers = torch.Tensor([.5 - (x_n**2**(i+1)) for i in range(m)].cuda()
    return torch.cat((x, powers))

def append_zeros(x, m):
    # querying function
    zeros = torch.zeros(m)
    return torch.cat((x, zeros))

class scale_query():
    def __init__(self, U, space_radius):
        # U and space_radius are real numbers.
        self.U = U
        self.space_radius = space_radius

    def __call__(self, x):
        return (self.U * x) / self.space_radius

