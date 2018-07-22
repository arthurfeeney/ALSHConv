

import torch

class ScaleUnder_U:
    # functor scales input magnitudes to be below U
    def __init__(self, U, device=torch.device('cuda')):
        self.U = torch.tensor(U).to(device)
        self.denom = torch.tensor(.0000001).to(device)
        self.device = device

    def __call__(self, input, update_denom=True):
        
        if update_denom:
            new_denom = input.view(input.size(0), -1).norm(dim=1).max()
            if new_denom > self.denom:
                self.denom = new_denom

        return self.U * input / self.denom


