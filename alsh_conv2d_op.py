
import torch
import pyinn as P

class ALSHConv2dOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, kernels, kernel_size, stride, padding,
                in_channels, Q, m, hf, table, table_size):
        # x should be a patch matrix
        # dimensions of the input
        input_dims = x.size()
        num_inputs = input_dims[0]
        h1 = input_dims[2]
        w1 = input_dims[3]

        # height and width of the output
        h2 = int((h1 - kernel_size + 2*padding) / stride + 1)
        w2 = int((w1 - kernel_size + 2*padding) / stride + 1)

        y = x.cuda()
        patches = P.im2col(y, kernel_size, stride, padding)

        # reform it as a 2d matrix.
        patch_matr = patches.view(kernel_size**2 * in_channels, -1)

        # get the average column from the input
        ave_patch_col = torch.mean(patch_matr.transpose(0,1), dim=0)

        # make the average column a unit vector
        unit_ave_col = ave_patch_col / torch.mean(ave_patch_col)

        # Hash the average unit column
        hash_out = hf(Q(unit_ave_col, m))
        hash_out.fmod_(table_size)
        hash_out.abs_()
        index = hash_out.long()

        # get the "active set" rows
        rows = torch.Tensor([]).long().cuda()#table[index].long().cuda()

        ctx.save_for_backward(patch_matr, kernels, rows)
        ctx.extra = h2, w2, in_channels, h1, w1, kernel_size, stride, padding

        # make the matrix containing the active kernels.
        active_kernels = []
        if rows.size() == torch.Size([0]):
            active_kernels = kernels
        else:
            active_kernels = kernels.index_select(0, rows)

        ctx.intermediate = kernels #active_kernels

        #out = torch.zeros(kernels.size()[0], patch_matr.size()[1]).cuda()

        #sub_out = active_kernels.mm(patch_matr)

        # fill the output.
        #for i, row in enumerate(rows):
        #    out[row] = sub_out[i]

        out = kernels.mm(patch_matr)

        out = out.view(num_inputs, -1, h2, w2)

        ctx.out_channels = out.size()[1]

        return out


    @staticmethod
    def backward(ctx, d_out):
        patch_matr, kernels, rows = ctx.saved_tensors
        H, W, in_channels, h1, w1, kernel_size, stride, padding = ctx.extra
        out_channels = ctx.out_channels

        active_kernels = ctx.intermediate

        d_out = d_out.view(out_channels, -1)

        d_out_active_rows = d_out.index_select(0, rows)

        if d_out_active_rows.size() == torch.Size([0]):
            d_out_active_rows = d_out

        d_patch_matr = active_kernels.transpose(0,1).mm(d_out_active_rows)

        d_active_kernels = patch_matr.mm(d_out_active_rows.transpose(0,1))

        d_kernels = torch.zeros(kernels.size()).cuda()

        for i, row in enumerate(rows):
            d_kernels[row] = d_active_kernels[i]

        d_x = d_patch_matr.view(-1, in_channels, h1, w1)

        return d_x, d_kernels, None, None, None, None, None, None, \
               None, None, None





