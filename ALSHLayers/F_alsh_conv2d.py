
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import pyinn as P
from Utility.cp_utils import count_votes, rehash_alsh_table
from torch.nn.modules.utils import _pair

class ALSHConv(nn.Module):

    def scale_kernels_under_U(self, kernels):
        r"""
        makes all rows, r, in kernels ||r|| <= U < 1.
        """
        U = .99
        # get index of greatest magnitue kernel.
        denom = kernels.view(kernels.size()[0], -1).norm(dim=1).max()

        return U * kernels / denom

    def init_table(self, kernels, hash, P, m, table_size, out_channels,
                   device=torch.device('cuda')):

        table_dims = torch.Size([table_size, hash.num_hashes*2*kernels.size()[0]])

        self.table = torch.Tensor(table_dims).to(device).long().fill_(0)

        self.table_row_lengths = \
            torch.Tensor(table_size).to(device).long().fill_(0)

        kernel_under_U = self.scale_kernels_under_U(kernels)

        # this applies the hash function passed in. Not Python's hash function.
        indices = hash(P(kernel_under_U, m, device).transpose(0,1))
        indices = indices.view(hash.num_hashes, -1).long().to(device)
        indices.fmod_(table_size)
        indices.abs_()

        for h in indices:
            i = torch.tensor(0).long().to(device)
            for idx in h:
                self.table[idx, self.table_row_lengths[idx]] = i
                self.table_row_lengths[idx] += 1
                i += 1


    def rehash_table(self, kernels, hash, P, m, table_size, index, rows,
                     device=torch.device('cuda')):

        threshold = hash.num_hashes * kernels.size()[0]

        if (self.table_row_lengths > threshold).sum() != 0:
            self.init_table(kernels, hash, P, m, table_size, kernels.size()[0],
                            device=device)


        else:
            self.table_row_lengths[index] = 0

            kernels_under_U = self.scale_kernels_under_U(kernels[rows])

            indices = hash(P(kernels_under_U, m, device).transpose(0,1), rows)
            indices = indices.view(hash.num_hashes, -1).long().to(device)
            indices.fmod_(table_size)
            indices.abs_()

            for h in indices:
                i = torch.tensor(0).long().to(device)
                for idx in indices:
                    self.table[idx, self.table_row_lengths[idx]] = rows[i]
                    self.table_row_lengths[idx] += 1
                    i += 1

    def get_active_set(self, kernels, input, hash, table_size, Q, m,
                       in_channels, kernel_size, stride, padding, dilation,
                       las=None, device=torch.device('cuda')):
        r"""
        Votes on the best bucket in self.table. Returns
        the subset of kernels, the index of the bucket in the table,
        and the indices of that subset.
        """

        # votes contains the votes for multiple hash functions
        votes = self.vote(input, hash, Q, m, in_channels, kernel_size,
                          stride, padding, dilation, las, device=device).view(-1)

        votes.fmod_(table_size)
        votes.abs_()

        count = torch.empty(hash.num_hashes, table_size).to(device).long()
        for c in range(hash.num_hashes):
            count[c] = count_votes(votes, table_size, device).argmax()

        # the argmax is the index of table that got the most votes.
        indices = count.argmax(dim=1)

        rows = None
        for index in indices:
            bucket = self.table[index, 0:self.table_row_lengths[index]]
            if rows is None:
                rows = bucket
            else:
                rows = torch.cat((rows, bucket))

        if rows.size() != torch.Size([0]):
            # sort rows if it is not empty.
            rows = rows.sort()[0].unique(sorted=True)

        active_set = kernels[rows]
        return active_set, indices, rows


    def vote(self, input, hash, Q, m, in_channels, kernel_size, stride,
             padding, dilation, las=None, device=torch.device('cuda')):
        r"""
        instead of reshaping input using im2col it, applies the hashes dot
        product to the input using a convolution. Q needs to append an 'image', of
        whatever it appends, to every 'image' in the input -> this requires m to
        be a certain size. Also need to prevent hash.a from being updated!
        """
        d = in_channels + (self.m / (kernel_size[0] * kernel_size[1]))

        kernel = hash.a_to_kernel(d, kernel_size)

        if las is not None:
            las_and_last = \
                torch.cat((las, torch.range(in_channels, d-1).long().to(device)))
            kernel = kernel[:,las_and_last]

        input_Q = Q(input, m, kernel_size, device=device)

        # input regions should be unit vectors if using stable_ditribution P
        # and Q! DARN!
        dotted = F.conv2d(input_Q, kernel, stride=stride,
                          padding=padding, dilation=dilation)

        dotted.transpose_(0,1)

        dotted = dotted.contiguous().view(hash.num_hashes, -1)

        add = dotted + hash.all_b().unsqueeze(1).expand_as(dotted)

        votes = torch.floor(add / hash.r).long()

        return votes.long()

class MultiHash:
    def __init__(self, which_hash, num_hashes, **kwargs):
        self.hashes = [which_hash(**kwargs) for _ in range(num_hashes)]
        self.num_hashes = num_hashes
        self.device = kwargs['device']
        self.r = kwargs['r']

    r"""
    a bunch of this stuff probably needs to be CUDA kernels eventually...
    """
    def __call__(self, x, rows=None):
        ret = None
        for h in self.hashes:
            if ret is None:
                ret = h(x, rows)
            else:
                ret = torch.cat((ret, h(x, rows)))

        return ret

    def all_b(self):
        return torch.Tensor([h.b for h in self.hashes]).to(self.device)

    def a_to_kernel(self, d, kernel_size):
        every_a = [h.a for h in self.hashes]
        flat = None
        for a in every_a:
            if flat is None:
                flat = a
            else:
                flat = torch.cat((flat, a))
        return flat.view(self.num_hashes, d, kernel_size[0], kernel_size[1])


class F_ALSHConv2d(nn.Conv2d, ALSHConv):
    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding, dilation, bias, table_size, which_hash, num_hashes, r,
                 m, P, Q, device=torch.device('cuda')):
        # init conv2d. ALSHConv has no init function
        super().__init__(in_channels, out_channels, kernel_size, stride,
                         padding=padding, dilation=dilation, bias=bias)

        self.device = device

        hash_dim = kernel_size**2*in_channels + m

        self._hash = MultiHash(which_hash, num_hashes, dim=hash_dim, r=r,
                               device=self.device)
        self._table_size = table_size
        self.m = m
        self.P = P # preprocessing function, must take batch of images
        self.Q = Q # query function, must take batch of images

        self.num_kernels = self.weight.size()[0]

        self.init_table(self.weight.to(device), self._hash, P, m, table_size,
                        out_channels, device=self.device)

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
        rows = torch.LongTensor(self._table_size).random_(0, self.num_kernels-1)
        return rows.to(self.device)

    def _generate_active_set(self, input, las):
        if las is None:
            kernels = self.weight
        else:
            kernels = self.weight[:,las]

        self.active_set, self.index, self.rows = \
            self.get_active_set(kernels, input, self._hash, self._table_size,
                                self.Q, self.m, self.in_channels,
                                self.kernel_size, self.stride, self.padding,
                                self.dilation, las, device=self.device)
        return kernels

    def forward(self, input, las=None):
        if not self.training:
            # if testing, just run through it, don't get active set!
            return F.conv2d(input, self.weight, self.bias, self.stride,
                            self.padding, self.dilation), None

        if self.training and self.rows is not None:
            # if training and rows exists (not first pass,) rehash
            if (self.index == -1).sum() == 0:
                # random indices were used, need to rehash everything.
                self.init_table(self.weight.to(self.device), self._hash, self.P,
                                self.m, self._table_size, self.out_channels,
                                device=self.device)
            else:
                self.rehash_table(self.weight.to(self.device), self._hash, self.P,
                              self.m, self._table_size, self.index, self.rows,
                              device=self.device)

        kernels = self._generate_active_set(input, las)

        if self.rows.size() == torch.Size([0]):
            # if the bucket is empty, use a random subset of kernels and
            # specify that everything should be rehashed (-1 flag).
            self.index = torch.tensor(-1).to(self.device)
            self.rows = self._random_rows().sort()[0]
            self.active_set = kernels[self.rows]

        out = F.conv2d(input, self.active_set, self.bias, self.stride,
                       self.padding, self.dilation)

        if self.training:
            scale = torch.tensor(float(self.rows.size()[0]) /
                                 self.out_channels).to(self.device)
            out /= scale


        # returns rows for the next layer's las paramater.
        return out, self.rows

