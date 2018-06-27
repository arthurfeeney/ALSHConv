
import torch
import torch.nn

# locality-sensitive hash family

class StableDistribution():
    def __init__(self, dim, r, device=torch.device('cuda')):
        self.r = torch.tensor(r).to(device)
        self.a = torch.empty(dim).to(device)
        torch.nn.init.normal_(self.a)
        self.b = torch.rand(1).to(device) * r # b in [0, r)
        self.device = device

    def __call__(self, x, las=None):
        dot = self.a @ x
        return torch.floor((dot + self.b) / self.r).to(self.device)
