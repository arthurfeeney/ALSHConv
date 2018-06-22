
import torch
import cupy
from string import Template

def _append_norm_powers(x, m):
    # x is just a 1d array
    x_n = torch.norm(x)
    powers = torch.Tensor([x_n**2**(i+1) for i in range(m)]).cuda()
    return torch.cat((x, powers))

def append_norm_powers(x, m):
    if x.dim() == 1:
        return _append_norm_powers(x, m)
    elif x.dim() == 2:
        num_rows = x.size()[0]

        x_ns = x.norm(dim=1)

        powers = torch.empty(num_rows, m).cuda()

        for i in range(m):
            for j, x_n in enumerate(x_ns):
                powers[j,i] = x_n**2**(i+1)

        return torch.cat((x, powers), 1)

    elif x.dim() == 4: # mini-batch of images
        batch_size = x.size()[0]
        return append_norm_powers(x.view(batch_size, -1), m)


def _append_halves(x, m):
    # x is a 1d array.
    halves = torch.Tensor(m).cuda().fill_(.5)
    return torch.cat((x, halves))

def append_halves(x, m, kernel_size=None):
    if x.dim() == 1:
        return _append_halves(x, m)
    elif x.dim() == 2:
        num_cols = x.size()[1]
        halves = torch.empty(m, num_cols).cuda().fill_(.5)
        return torch.cat((x, halves), 0)

    elif x.dim() == 4:
        height, width = x.size()[2:]
        depth = m / (kernel_size[0] * kernel_size[1])

        halves = torch.empty(x.size()[0], depth,
                             height, width).cuda().fill_(.5)

        return torch.cat((x, halves), 1)



