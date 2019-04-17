

import torch


def scale_magnitudes_under_U(data, U):
    r"""
    For alsh to work, data must be <= U < 1. Scaling down does not affect
    the relative magnitudes of vectors, so the argmax remains the same. 
    """
    biggest = torch.norm(data, dim=1).max()
    data /= biggest / U
    return data
