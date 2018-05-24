
import torch

class ALSHOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, Q, m, hf, table, table_size, conv):

        ave_input = None
        if conv:
            ave_input = torch.mean(x.transpose(0,1), dim=0)
        else:
            ave_input = torch.mean(x, dim=0)

        # Query should be a unit vector when hashing.
        # It's okay to just normalize right before because it won't affect
        # relative ordering.

        unit_ave_input = ave_input / torch.norm(ave_input)

        print(unit_ave_input.size())


        hash_out = hf(Q(unit_ave_input, m))

        hash_out.fmod_(table_size)
        hash_out.abs_
        index = hash_out.int()

        rows = table[index].cuda()

        ctx.save_for_backward(x, weight, rows)

        # if rows is empty, just use the entire weight matrix. (this is ugly)
        matr = []
        if rows.size() == torch.Size([0]):
            matr = weight
        else:
            matr = weight.index_select(0, rows)

        ctx.intermediate = matr

        out = torch.zeros(x.size()[0], weight.size()[0]).cuda()

        # y = x A.transpose
        sub_out = x.mm(matr.transpose(0,1)).cuda()

        for i, row in enumerate(rows):
            out[:,row] = sub_out[:,i]

        return out

    @staticmethod
    def backward(ctx, d_out):
        x, weight, rows = ctx.saved_tensors

        matr = ctx.intermediate
        d_out_active_rows = d_out.index_select(1, rows)

        # if rows was empty, then the entire matrix was used. So use all of d_out
        if d_out_active_rows.size() == torch.Size([0]):
            d_out_active_rows = d_out

        # dx = dy A
        d_x = d_out_active_rows.mm(matr)

        # dA = x.transpose dy
        d_matr = x.transpose(0,1).mm(d_out_active_rows)

        d_weight = torch.zeros(weight.size()).cuda()

        for i, row in enumerate(rows):
            d_weight[row] = d_matr[:,i]

        return d_x, d_weight, None, None, None, None, None
