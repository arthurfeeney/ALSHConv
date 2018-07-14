
import torch
from Hash.norm_and_halves import append_halves

def _append_sub_norm_powers(x, m, device=torch.device('cpu')):
    # x is just a 1d array
    x_n = torch.norm(x)
    powers = torch.Tensor([.5 - x_n**2**(i+1) for i in range(m)]).to(device)
    return torch.cat((x, powers))

def append_sub_norm_powers(x, m, U=0, check=True, kernel_size=None,
                           device=torch.device('cpu')):
    if x.dim() == 1:
        return _append_sub_norm_powers(x, m, device)
    elif x.dim() == 2:
        num_rows = x.size()[0]

        x_ns = x.norm(dim=1)

        powers = torch.empty(num_rows, m).to(device)

        for i in range(m):
            for j, x_n in enumerate(x_ns):
                powers[j,i] = .5 - x_n**2**(i+1)

        powers[powers == float('inf')] = 0

        return torch.cat((x, powers), 1)

    elif x.dim() == 4: # mini-batch of images
        if check:
            batch_size = x.size()[0]
            return append_sub_norm_powers(x.view(batch_size, -1), m, device)
        

        # CAN'T compute P for each of the regions on input to convolution.
        # Since m is so large, most values will be near .5. 
        # Using U provides a rough way to approximate it. 

        #depth = int(m / kernel_size[0] * kernel_size[1])
        #height, width = x.size()[2:]

        #powers = torch.empty(x.size(0), 1, height, width).to(device)


        #U_fake = torch.Tensor([.5 - U**2**i+1 for i in range(height*width)])
        #U_fake = U_fake.view(height, width)

        #print(powers.size(), x.size())


        #for i in range(powers.size(0)):
        #     powers[i,0,:,:] = U_fake#.expand(depth, height, 
                                            #      width).to(device)

        #return torch.cat((x, powers), 1)
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



