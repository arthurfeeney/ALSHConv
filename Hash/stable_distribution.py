
import torch
import torch.nn

# locality-sensitive hash family

class StableDistribution():
    def __init__(self, dim, r):
        self.r = r
        self.a = torch.empty(dim).cuda()
        torch.nn.init.normal_(self.a)
        self.b = torch.rand(1).cuda() * r # b in [0, r)

    def __call__(self, x):
        dot = self.a.dot(x)
        return torch.floor((dot + self.b) / self.r).cuda()
