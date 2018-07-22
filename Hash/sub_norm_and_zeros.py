
import torch
from Hash.norm_and_halves import append_halves
from Utility.cp_utils import fill_powers_of_norms


def _append_sub_norm_powers(x, m, device=torch.device('cuda')):
    # x is just a 1d array
    x_n = torch.norm(x)
    powers = fill_powers_of_norm(x_n, m, sub=True, device=device)
    return torch.cat((x, powers))

def append_sub_norm_powers(x, m, U=0, check=True, kernel_size=None,
                           device=torch.device('cuda')):
    if x.dim() == 1:
        return _append_sub_norm_powers(x, m, device)
    elif x.dim() == 2:
        num_rows = x.size()[0]

        x_ns = x.norm(dim=1)

        powers = fill_powers_of_norms(x_ns, m, sub=True, 
                                      device=torch.device('cpu'))

        return torch.cat((x, powers), 1)

    elif x.dim() == 4: # mini-batch of images
        if check:
            batch_size = x.size()[0]
            return append_sub_norm_powers(x.view(batch_size, -1), m, device)
        
        return append_halves(x, m, kernel_size, device=device)



def _append_zeros(x, m, device=torch.device('cuda')):
    # x is a 1d array.
    halves = torch.Tensor(m).to(device).fill_(0)
    return torch.cat((x, halves))

def append_zeros(x, m, kernel_size=None, device=torch.device('cuda')):
    if x.dim() == 1:
        return _append_zeros(x, m, device)
    elif x.dim() == 2:
        num_cols = x.size()[1]
        halves = torch.empty(m, num_cols).to(device).fill_(0)
        return torch.cat((x, halves), 0)

    elif x.dim() == 4:
        height, width = x.size()[2:]
        depth = m / (kernel_size[0] * kernel_size[1])

        halves = torch.empty(x.size(0), depth,
                             height, width).to(device).fill_(0)

        return torch.cat((x, halves), 1)



