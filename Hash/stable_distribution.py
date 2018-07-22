
import torch
import torch.nn.functional as F

# locality-sensitive hash family

class StableDistribution():
    def __init__(self, dim, r, device=torch.device('cuda')):
        self.r = torch.tensor(r).to(device)
        self.a = torch.empty(1, dim).to(device)
        torch.nn.init.normal_(self.a)
        self.b = torch.rand(1).to(device) * r # b in [0, r)
        self.device = device

    def __call__(self, x, las=None, **kwargs):

        if x.dim() == 4:
            # if x is input to convolution
            dotted = F.conv2d(x, weight=kwargs['kernel'],
                                 stride=kwargs['stride'],
                                 padding=kwargs['padding'],
                                 dilation=kwargs['dilation'])        

            return torch.floor((dotted + self.b) / self.r).to(self.device)
    


        # used when x is a matrix
        dot = self.a @ x
        x = torch.floor((dot + self.b) / self.r).to(self.device)
        return x
