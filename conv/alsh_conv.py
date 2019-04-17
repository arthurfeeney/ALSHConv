import sys
sys.path.append('../lsh/')

import torch
from lsh.tables_cpu import TablesCPU
from lsh.tables_gpu import TablesGPU
#from Conv._ext import get_active_set


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class ALSHConv:
    def init_ALSH(self, num_tables, final_num_tables, table_size, which_hash, hash_init_params,
                  num_hashes, dim, num_filters, device):
        assert isinstance(device, str), 'ALSHConv, device must be a string'
        assert device == 'cpu' or device == 'gpu', \
            'ALSHConv, device must be a \'cpu\' or \'gpu\'.'

        self.tables = TablesCPU(num_tables, table_size, which_hash,
                                hash_init_params, num_hashes, dim)

        self.final_num_tables = final_num_tables

        self.num_filters = num_filters

        self.bucket_stats = AverageMeter()

    def fill_table(self, filters):
        num = filters.size(0)
        self.tables.insert_data(filters.view(num, -1),
                                torch.arange(0, num).long())

    def refill(self, active_kernels, active_set, table_indices):
        self.tables.clear_row(table_indices)
        num = active_kernels.size(0)
        self.tables.insert_data(active_kernels.view(num, -1), active_set)

    def trim(self):
        if self.tables.num_tables > self.final_num_tables:
            self.tables.trim()

    def most_freq(self, x, k):
        r'''
        finds the k most frequently occuring values in x
        '''
        bins = self.tables.table_size
        item_freq = torch.histc(x.cpu(), bins=bins, max=bins)
        self.bucket_stats.update(item_freq)
        return item_freq.topk(k)[1]

    def get_active_set(self,
                       input,
                       kernel_size,
                       stride,
                       padding,
                       dilation,
                       LAS=None):

        ti = self.tables.get(input,
                             kernel_size=kernel_size,
                             stride=stride,
                             padding=padding,
                             dilation=dilation,
                             LAS=LAS)

        ti = ti.view(self.tables.num_tables, -1)

        k = 5

        topkl = torch.zeros(ti.size(0), k).long()

        for row in range(ti.size(0)):
            topkl[row] = self.most_freq(ti[row], k=k)

        AS = torch.LongTensor([])
        for i in torch.arange(0, self.tables.num_tables).long():
            for j in topkl[i]:
                ids = torch.LongTensor(self.tables.tables[i][j])
                AS = torch.cat((AS, ids)).unique(sorted=False)

        return AS.sort()[0], ti
