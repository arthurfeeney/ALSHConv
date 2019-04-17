import torch
import torch.nn as nn
import collections
from statistics import mode


class TablesCPU:
    def __init__(self, num_tables, table_size, which_hash, hash_init_params,
                 num_hashes, dim):
        r"""
        These tables are intended for ALSH on a cpu.

        Takes integral values for:
            num_tables, which_hash, num_hashes table_size, init_row_len
        hash_init_params is a dict of parameters needed for the hashes
        dtype is a torch.Tensor value type
        device must be a torch.device()

        which_hash must be a hash funtion that returns a value > 0
        and can handle a vector, matrix, imageBatch.
        """

        self.num_tables = num_tables
        self.table_size = table_size
        self.num_hashes = num_hashes
        self.dim = dim

        t = range(num_tables)
        self.hashes = [
            which_hash(num_hashes, dim, hash_init_params) for _ in t
        ]
        # tables does not contain keys. Only values.
        self.tables = [[[] for _ in range(table_size)] for _ in t]

    def trim(self, count=4):
        '''
        removes last count tables from the list of tables.
        This may be useful for training ALSHConv2d by making the replacements
        a little slower initially. 
        '''
        if len(self.tables) > 3:
            del self.hashes[-count:]
            del self.tables[-count:]
            self.num_tables -= count 
            print(self.num_tables)
            return True
        return False


    def get(self, key, **kwargs):
        return torch.stack([
            hash.query(key, **kwargs) % self.table_size for hash in self.hashes
        ])

    def put(self, key, **kwargs):
        return [
            hash.pre(key, **kwargs) % self.table_size for hash in self.hashes
        ]

    def insert(self, key, value):
        r"""
        x must be some vector type with a length == self.dim
        This function inserts x into each table in self.tables.
        """
        rows = self.pre(key)
        for ti in torch.arange(0, self.num_tables):
            self.tables[ti][rows[ti]].append(value)

    def insert_data(self, keys, values):
        r"""
        inserts a sequence of values into tables based on keys.
        keys[i] is the key for values[i]
        For this application, only values need to be stored in the hash table.
        They work as references to the keys.
        """

        t = range(self.num_tables)
        self.tables = [[[] for _ in range(self.table_size)] for _ in t]
        rows = self.put(keys)

        # can parallelize outer loop for each table?

        for ti in torch.arange(0, self.num_tables).long():
            #for val in values:#torch.arange(0, len(values)):
            for row, val in zip(rows[ti], values):
                self.tables[ti][row].append(int(val))

    def get_query_rows(self, q):
        indices = self._get(q)
        rows = [None] * self.num_tables
        for ti in torch.arange(0, self.num_tables):
            rows[ti] = self.tables[ti][indices[ti]]
        return rows

    def clear_row(self, row_indices):
        r"""
        clears all rows in the sequence row_indices
        row_indices[0] is the index of the row to clear in the first table.
        [1,2,3] clears rows 1, 2, and 3 from tables 0, 1, and 2
        """
        for ti in torch.arange(0, self.num_tables).long():
            for row in row_indices[ti]:
                for r in row.long():
                    self.tables[ti][r] = []
            #self.tables[ti][row_indices[ti]] = torch.Tensor([]).long()
