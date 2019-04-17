
import sys
sys.path.append('../')

from LSH.multi_hash_srp import MultiHash_SRP
import torch


h = MultiHash_SRP(3, 36) # 27 = 3x3x3 


input = torch.Tensor(5, 3, 3, 3).normal_()

hashes = h.query(input, kernel_size=3, stride=1, padding=1, dilation=1)

print(hashes) # should return three numbers
