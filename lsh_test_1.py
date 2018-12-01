

from LSH.tables_cpu import TablesCPU
from LSH.multi_hash_srp import MultiHash_SRP

import torch

t = TablesCPU(2, table_size=10, which_hash=MultiHash_SRP, hash_init_params={}, 
              num_hashes=3, dim=2)

t.insert(torch.Tensor([2.0,2.0]), 0)


rows = t.get_query_rows(torch.Tensor([2.0,2.0]))


# Output:
# [0]
# [0]
# End Output
for i in range(2):
    print(next(rows))
