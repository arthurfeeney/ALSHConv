
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import pyinn as P
from Utility.cp_utils import count_votes, rehash_alsh_table, get_unique
from torch.nn.modules.utils import _pair

import time

class ScaleUnder_U:
    # functor makes input magnitudes < U
    def __init__(self, U):
        self.U = U

    def __call__(self, input, update_denom=True):
        
        if update_denom:
            self.denom = input.view(input.size(0), -1).norm(dim=1).max()
        
        return self.U * input / self.denom


class ALSHConv(nn.Module):
    def init_table(self, kernels, hash, Q, P, m, U, table_size, out_channels,
                   device=torch.device('cuda')):


        row_len = hash.num_hashes*2*kernels.size(0)
        table_dims = torch.Size([table_size, row_len])

        self.table = torch.Tensor(table_dims).to(device).long().fill_(0)

        self.table_row_lengths = \
            torch.Tensor(table_size).to(device).long().fill_(0)

        self.scale_func = ScaleUnder_U(U)

        kernel_under_U = self.scale_func(kernels)

        indices = hash(Q(P(kernel_under_U, m, self.scale_func.U, 
                           device=device).transpose(0,1), 
                         m, device=device))
        indices = indices.view(hash.num_hashes, -1).long().to(device)
        indices.fmod_(table_size)
        indices.abs_()

        for h in indices:
            i = torch.tensor(0).long().to(device)
            for idx in h:
                #if self.table[idx][self.table[idx] == i].numel() == 0 or i==0:
                # add it to the row if it isn't in it already
                self.table[idx, self.table_row_lengths[idx]] = i
                self.table_row_lengths[idx] += 1
                i += 1

    def rehash_table(self, kernels, hash, P, m, table_size, index, rows,
                     device=torch.device('cuda')):

        threshold = hash.num_hashes * kernels.size(0)

        if (self.table_row_lengths > threshold).sum() != 0:
            self.init_table(kernels, hash, P, m, table_size, 
                            kernels.size(0), device=device)


        else:
            self.table_row_lengths[index] = 0

            kernels_under_U = self.scale_func(kernels[rows])

            kernels_under_U = P(kernels_under_U, m, self.scale_func.U,
                                 device=device).transpose(0,1)

            kernels_under_U = Q(kernels_under_U, m, device=device)

            indices = hash(kernels_under_U, rows)
            indices = indices.view(hash.num_hashes, -1).long().to(device)
            indices.fmod_(table_size)
            indices.abs_()

            for h in indices:
                i = torch.tensor(0).long().to(device)
                for idx in h:
                    self.table[idx, self.table_row_lengths[idx]] = \
                        rows[i]
                    self.table_row_lengths[idx] += 1
                    i += 1

    def get_table_row(self, index):
        return self.table[index, 0:self.table_row_lengths[index]]

    def get_active_set(self, kernels, input, hash, table_size, Q, P, m,
                       in_channels, kernel_size, stride, padding, dilation,
                       las=None, device=torch.device('cuda')):
        r"""
        Votes on the best bucket in self.table. Returns
        the subset of kernels, the index of the bucket in the table,
        and the indices of that subset.
        """

        # votes contains the votes for multiple hash functions
        votes = self.vote(input, hash, Q, P, m, in_channels, kernel_size,
                          stride, padding, dilation, las, 
                          device=device)


        #indices = torch.stack([\
        #            count_votes(votes[c].view(-1), table_size, 
        #            device) \
        #            for c in range(votes.size(0))\
        #          ]).to(device)

        #indices = indices.sum(dim=0)

        #print(indices.size())

        # votes : batch_size, num_hashes, region, region
        # vote_for_hash: batch_size num_hashes, num_table
        # indices: num_hashes
        
        votes_for_hash = torch.stack([count_votes(votes[c][h], table_size,
                                                 device=device) for \
                                     c in range(votes.size(0)) for h \
                                        in range(votes.size(1))])

        print(votes_for_hash.size())
        
        indices = votes_for_hash.sum(dim=0).sum(dim=1) 
        print(indices)


        rows = torch.cat([self.get_table_row(index) for index in indices])

        if rows.size() != torch.Size([0]):
            # sort rows if it is not empty.
            rows = get_unique(rows.sort()[0], device)

        active_set = kernels[rows]
        return active_set, indices, rows


    def vote(self, input, hash, Q, P, m, in_channels, kernel_size, stride,
             padding, dilation, las=None, device=torch.device('cuda')):
        r"""
        instead of reshaping input using im2col it, applies the hashes dot
        product to the input using a convolution. Q needs to append an 
        'image', of whatever it appends, to every 'image' in the input -> 
        this requires m to be a certain size. Also need to prevent hash.a 
        from being updated!
        """
        d = in_channels + (self.m*2 / (kernel_size[0] * kernel_size[1]))

        kernel = hash.a_to_kernel(d, kernel_size, m)

        if las is not None:
            m_plane = torch.range(in_channels, d-1).long().to(device)
            las_and_last = torch.cat((las, m_plane))
            kernel = kernel[:,las_and_last]

    
        # P(Q(T(q))) removes dependence on norm of query.
        # P is approximated because you can't append to each region. 
        input_Q = Q(self.scale_func(input, update_denom=False), m, 
                    kernel_size, device=device)

        input_Q = P(input_Q, m, U = self.scale_func.U, check=False, 
                    kernel_size=kernel_size, device=device)

        dotted = F.conv2d(input_Q, kernel, stride=stride,
                          padding=padding, dilation=dilation)        

        return (dotted > 0).to(device).long()

class MultiHash:
    def __init__(self, which_hash, num_hashes, **kwargs):
        self.hashes = [which_hash(**kwargs) for _ in range(num_hashes)]
        self.num_hashes = num_hashes
        self.device = kwargs['device']


    r"""
    a bunch of this stuff probably needs to be CUDA kernels eventually...
    """
    def __call__(self, x, rows=None):
        return torch.stack([h(x, rows) for h in self.hashes]).to(self.device)

    def all_b(self):
        return torch.Tensor([h.b for h in self.hashes]).to(self.device)

    def a_to_kernel(self, d, kernel_size, m):
        flat = torch.stack([h.a for h in self.hashes]).to(self.device)
        return flat.view(self.num_hashes, d, kernel_size[0], kernel_size[1])


class F_ALSHConv2d(nn.Conv2d, ALSHConv):
    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding, dilation, bias, table_size, which_hash, 
                 num_hashes, m, U, P, Q, device=torch.device('cuda'), 
                 **kwargs):
        # init conv2d. ALSHConv has no init function
        super().__init__(in_channels, out_channels, kernel_size, stride,
                         padding=padding, dilation=dilation, bias=bias)

        self.device = device

        hash_dim = kernel_size**2*in_channels + 2*m

        self._hash = MultiHash(which_hash, num_hashes, dim=hash_dim,
                               device=self.device, **kwargs)
        self._table_size = table_size
        self.m = m
        self.P = P # preprocessing function, must take batch of images
        self.Q = Q # query function, must take batch of images
        self.U = U

        self.num_kernels = self.weight.size(0)

        self.init_table(self.weight.to(device), self._hash, Q, P, m, U,
                        table_size, out_channels, device=self.device)

        self.active_set, self.index, self.rows = None, None, None

    def _apply_hash(self, x):
        r"""
        returns the output of every hash function in a single 1-d list.
        If each hash function returns a list then they are concatenated.
        """
        applied = []
        for h in self._hash:
            applied += h(x)
        return applied

    def _random_rows(self):
        lower, upper = 0, self.num_kernels-1
        rows = torch.LongTensor(self._table_size).random_(lower, upper)
        return get_unique(rows.sort()[0].to(self.device), self.device)

    def _generate_active_set(self, input, las):
        if las is None:
            kernels = self.weight
        else:
            kernels = self.weight[:,las]

        self.active_set, self.index, self.rows = \
            self.get_active_set(kernels, input, self._hash, self._table_size,
                                self.Q, self.P, self.m, self.in_channels,
                                self.kernel_size, self.stride, self.padding,
                                self.dilation, las, device=self.device)

    def _empty_row_case(self, las):
        # if the bucket is empty, use a random subset of kernels and
        # specify that everything should be rehashed (-1 flag).
        self.index = torch.tensor(-1).to(self.device)
        self.rows = get_unique(self._random_rows().sort()[0], self.device)
        kernels = self.weight[:,las].to(device)
        self.active_set = kernels[self.rows]

    def _get_scale(self):
        #num = float(self.rows.size(0))
        #s = torch.tensor(num / self.out_channels).to(self.device)
        num = self._hash.num_hashes
        denom = self._table_size
        return torch.tensor(num / denom).to(self.device)

    def forward(self, input, las=None):
        r"""
        takes input and the last layers active set. 
        if las is None, all kernels channels are used. 
        Otherwise, only channels in las are used.
        """
        #if not self.training:
            # if testing, just run through it, don't get active set!
        #    return F.conv2d(input, self.weight, self.bias, self.stride,
        #                    self.padding, self.dilation), None

        #start = time.time()
        
        if self.training and self.rows is not None:
            # if training and rows exists (not first pass,) rehash
            if (self.index == -1).sum() == 0:
                # random indices were used, need to rehash everything.
                self.init_table(self.weight.to(self.device), self._hash, 
                                self.Q, self.P, self.m, self.U, 
                                self._table_size, self.out_channels, 
                                device=self.device)
            else:
                self.rehash_table(self.weight.to(self.device), self._hash, 
                                  self.P, self.m, self._table_size, 
                                  self.index, self.rows, device=self.device)

        #rehash_time = time.time() - start

        #start = time.time()
        self._generate_active_set(input, las)
        #get_as_time = time.time() - start

        if self.rows.size(0) == 0:
            self._empty_row_case(las)

        #start = time.time()
        output = F.conv2d(input, self.active_set, self.bias, self.stride,
                       self.padding, self.dilation)
        #conv_time = time.time() - start


        #print('rehash: ', rehash_time)
        #print('get as: ', get_as_time)
        #print('conv time: ', conv_time)
        #print('total: ', rehash_time + get_as_time + conv_time)

        output /= self._get_scale()

        # returns rows for the next layer's las paramater.
        #print(output.size())
        return output, self.rows

