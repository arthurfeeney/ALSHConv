
import torch
import torch.nn as nn
import torch.nn.functional as F
from Utility.cp_utils import count_votes, rehash_alsh_table, get_unique
from Hash.scale_under_u import ScaleUnder_U

import time

class ALSHConv(nn.Module):
    def init_table(self, kernels, hash, Q, P, m, U, table_size, out_channels,
                   device=torch.device('cuda')):


        row_len = hash.num_hashes*hash.bits*2*kernels.size(0)
        table_dims = torch.Size([table_size, row_len])

        self.table = torch.Tensor(table_dims).to(device).long().fill_(0)

        self.table_row_lengths = \
            torch.Tensor(table_size).to(device).long().fill_(0)

        self.scale_func = ScaleUnder_U(U, device=device)
        kernel_under_U = self.scale_func(kernels)

        input_P = P(kernel_under_U, m, self.scale_func.U, device=device)
        #input_P = input_P.to(device)

        #input_Q = Q(input_P.transpose(0,1), m, device=device)

        indices = hash(input_P.transpose(0,1))

        indices = indices.view(hash.num_hashes, -1).long()
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

            #kernels_under_U = Q(kernels_under_U, m, device=device)

            indices = hash(kernels_under_U.transpose(0,1), rows)
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

        votes = votes.view(hash.num_hashes, -1)
 
        tallied = count_votes(votes, table_size, device=device)
        
        indices = tallied.argmax(1)

        print(indices)

        #rows = torch.cat([self.get_table_row(index) for index in indices])
        rows = self.table[indices].view(-1)

        if rows.size() != torch.Size([0]):
            # sort rows if it is not empty.
            rows = get_unique(rows.sort()[0], sorted=False, device=device)

        active_set = kernels[rows]
        return active_set, indices, rows


    def vote(self, input, hash, Q, P, m, in_channels, kernel_size, stride,
             padding, dilation, las=None, device=torch.device('cuda')):
        r"""
        instead of reshaping input using im2col it, applies the hashes dot
        product to the input using a convolution. Q needs to append an 
        'image', of whatever it appends, to every 'image' in the input -> 
        this requires m to be a certain size.
        """

        start = time.time()

        d = in_channels + (self.m / (kernel_size[0] * kernel_size[1]))
        d = torch.tensor(d).to(device).long()

        print(d, m, hash.hashes[0].a.size())

        kernel = hash.a_to_kernel(d, kernel_size, m)

        if las is not None:
            m_plane = torch.range(in_channels, d).long().to(device)
            las_and_last = torch.cat((las, m_plane))
            kernel = kernel[:,las_and_last]
        
        input = Q(input, m, kernel_size, device=device)


        p = time.time()
        dotted = hash(input, kernel=kernel, stride=stride, padding=padding,
                      dilation=dilation)
        q = time.time()-start
        print('hash time', q)

                 
        elapsed = time.time() - start

        print('vote time ', elapsed)

        return dotted

        
