
import torch


def num_digits(n):
    if n == 0:
        return torch.tensor(1).long()
    return torch.tensor(torch.floor(torch.log10(n)) + 1).long()

def tensor_to_int(input, device=torch.device('cuda')):
    if device == torch.device('cpu'):
        # Not as concerned with CPU performance.
        if input.dim() == 1:
            return _1d_tensor_to_int_CPU(input)
        elif input.dim() == 2:
            return _2d_tensor_to_int_CPU(input)

def _1d_tensor_to_int_CPU(input):
    r"""
    Concatenates elements of a LongTensor to a single long.
    """
    assert input.dim() == 1, '_1d_tensor_to_int_CPU, input.dim != 1'
    out = 0
    for num in input.long():
        #num_dig = len(str(num.long().item()))
        num_dig = num_digits(num.float())
        out *= 10**num_dig
        out += num
    return out.long()

def _2d_tensor_to_int_CPU(input):
    r"""
    each row of the input tensor is concantenated to a single int.
    Output is a LongTensor.
    """
    assert input.dim() == 2, '_2d_tensor_to_int_CPU, input.dim != 2'
    out = torch.Tensor(input.size(0)).long()
    for i in range(input.size(0)):
        out[i] = _1d_tensor_to_int_CPU(input[i])
    return out

