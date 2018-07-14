
import torch



# should compute H(P(T(X)))
# and H(Q(T(X))), T makes it so query doesn't need to be a unit vector.

class SignRandomProjection():
    def __init__(self, dim, device=torch.device('cuda')):
        self.a = torch.empty(dim).to(device)
        torch.nn.init.normal_(self.a)
        self.device=device

    def __call__(self, x, las=None):
        return (self.a @ x > 0).to(self.device).long()
