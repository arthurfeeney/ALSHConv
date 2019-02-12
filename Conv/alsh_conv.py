

import sys
sys.path.append('../LSH/')

import torch
from LSH.tables_cpu import TablesCPU
from LSH.tables_gpu import TablesGPU
from Conv._ext import get_active_set

class ALSHConv:
    def init_ALSH(self, num_tables, table_size, which_hash, hash_init_params,
                  num_hashes, dim, num_filters, device):
        assert isinstance(device, str), 'ALSHConv, device must be a string'
        assert device == 'cpu' or device == 'gpu', \
            'ALSHConv, device must be a \'cpu\' or \'gpu\'.'

        self.tables = TablesCPU(num_tables, table_size, which_hash,
                                    hash_init_params, num_hashes, dim)
        #self.tables = TablesGPU(num_tables, table_size, which_hash,
        #                        hash_init_params, num_hashes, dim, num_filters)
        self.num_filters = num_filters

    def fill_table(self, filters):
        num = filters.size(0)
        self.tables.insert_data(filters.view(num, -1),
                                torch.arange(0, num).long())

    def refill(self, active_kernels, active_set, table_indices):
        self.tables.clear_row(table_indices)
        num = active_kernels.size(0)
        self.tables.insert_data(active_kernels.view(num, -1), active_set)

    def most_freq(self, x, k):
        r'''
        finds the k most frequently occuring values in x
        '''
        bins = self.tables.table_size
        item_freq = torch.histc(x.cpu(), bins=bins, max=bins)
        return item_freq.topk(k)[1]

    def get_active_set(self, input, kernel_size, stride, padding, dilation):
        ti = self.tables.get(input, kernel_size=kernel_size, stride=stride,
                             padding=padding, dilation=dilation)

        ti=ti.view(self.tables.num_tables, -1)

        k = 5

        topkl = torch.zeros(ti.size(0), k).long()
        #get_active_set.k_freq_buckets(ti.cpu(), k, self.tables.table_size,
        #                              topkl)

        for row in range(ti.size(0)):
            topkl[row] = self.most_freq(ti[row], k=k)

        #
        # This may stay as for loops because tables is a python list?
        # Kind of finicky for pytorch extension
        #
        AS = torch.LongTensor([])
        for i in torch.arange(0, self.tables.num_tables).long():
            for j in topkl[i]:
                AS = torch.cat((AS, torch.LongTensor(self.tables.tables[i][j]))).unique(sorted=False)

        return AS.sort()[0], ti
