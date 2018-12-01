

from LSH.tables_cpu import TablesCPU
from LSH.multi_hash_srp import MultiHash_SRP

import torch
import time


count = 10000
dim = 100


keys = [torch.randn(dim) for _ in range(count)]
biggest = max([a.norm() for a in keys])
keys = [.75 * a / biggest for a in keys]
values = [i for i in range(count)]


t = TablesCPU(60, table_size=100, which_hash=MultiHash_SRP, 
              hash_init_params={}, num_hashes=64, dim=dim)




t.insert_data(keys, values)


rows = t.get_query_rows(torch.Tensor([0.3]*dim))

print(rows.mode)

rows = list(rows)
# Output:
# End Output
for i in range(len(rows)):
    print(rows[i])
