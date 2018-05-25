
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from alsh_op import ALSHOp

class ALSHLinear(nn.Module):
    def __init__(self, input_size, num_nodes, hf, table_size, m, P, Q):
        super(ALSHLinear, self).__init__()

        self.m = m
        self.P = P # Preprocessing function
        self.Q = Q # Query function

        self.output_size = num_nodes

        self.weight = nn.Parameter(
                        torch.zeros(num_nodes, input_size),
                        requires_grad=True).cuda()
        nn.init.xavier_normal_(self.weight)

        self.__hash = hf # should be a class that defines __call__
        self.__table_size = table_size


        self.table = self.__build_alsh_table__()

        self.cache = None


    def __build_alsh_table__(self):
        table = [None]*self.__table_size
        for i, row in enumerate(self.weight):
            hash_out = self.__hash(self.P(row, self.m))
            index = (torch.abs(hash_out % self.__table_size)).long().cuda()
            if table[index] is not None:
                table[index].append(i)
            else:
                table[index] = [i]

        for i in range(len(table)):
            table[i] = torch.Tensor(table[i]).long().cuda()

        return table

    def rehash(self):
        ave_input = torch.mean(self.cache, dim=0)

        # query needs to be normalized for some ALSH.
        # it doesn't affect relative ordering of q.transpose x, so
        # it can just be done right before using Q.
        unit_ave_input = ave_input / torch.norm(ave_input)

        hash_out = self.__hash(self.Q(unit_ave_input, self.m))
        hash_out.fmod_(self.__table_size)
        hash_out.abs_()

        table_index = hash_out.long().cuda()
        rows = self.table[table_index]

        self.table[table_index] = torch.Tensor([]).long().cuda()

        self.table = [a.tolist() for a in self.table]

        rows = rows.split(1)

        for row in rows:
            hash_out2 = self.__hash(self.P(self.weight[row][0], self.m))
            hash_out2.fmod_(self.__table_size)
            hash_out2.abs_()
            index = hash_out2.long().cuda()
            self.table[index].append(row) # This may be run on cpu? :(

        self.table = [torch.Tensor(a).long().cuda() for a in self.table]

    def forward(self, x, mode):
        if (self.cache is not None) and mode:
            # if there is something from the last pass (it isn't the first)
            # and it is in training mode, then rehash weights!
            self.rehash()

        out = ALSHOp.apply(x, self.weight, self.Q, self.m, self.__hash,
                           self.table, self.__table_size).cuda()

        self.cache = x

        return out

