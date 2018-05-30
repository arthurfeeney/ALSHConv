
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
        h2 = (h1 - kernel_size + 2*padding) // stride + 1
        w2 = (w1 - kernel_size + 2*padding) // stride + 1

        y = x.cuda()
        patches = P.im2col(y, kernel_size, stride, padding)

        ctx.patch_size = patches.size()

        # change ordering so columns correspond to kernel regions when
        # viewed - permute just does transposes. transposes just
        # change indexing method so not slow.
        patches = patches.permute(1,2,3,0,4,5).contiguous()

        # reform it as a 2d matrix.
        patch_matr = patches.view(kernel_size**2 * in_channels, -1)

        #ave_im_patch = torch.zeros(patch_matr[:,:h2*w2].size())

        #for i in range(num_inputs):
        #    ave_im_patch += patch_matr[:,i*h2*h2:(i+1)*h2*w2]

        #ave_im_patch /= num_inputs

        # get the average column from the input
        #sum_patch_col = patch_matr.transpose(0,1).sum(0)
        #max_each_row_vect = patch_matr.transpose(0,1).max(0)[0]

        # make the average column a unit vector
        #unit_row_vect = max_each_row_vect / torch.norm(max_each_row_vect)

        # Hash the average unit column
        #hash_out = hf(Q(unit_ave_col, m))
        #hash_out.fmod_(table_size)
        #hash_out.abs_()
        #index = hash_out.long()

        # get the "active set" rows
        rows = torch.Tensor([]).long().cuda()#table[index].long().cuda()

        ctx.save_for_backward(patch_matr, kernels, rows)
        ctx.extra = h2, w2, in_channels, h1, w1, kernel_size, stride, padding

        # make the matrix containing the active kernels.
        #active_kernels = []
        #if rows.size() == torch.Size([0]):
        #    active_kernels = kernels
        #else:
        #    active_kernels = kernels.index_select(0, rows)

        ctx.intermediate = kernels #active_kernels

        #out = torch.zeros(kernels.size()[0], patch_matr.size()[1]).cuda()

        #sub_out = active_kernels.mm(patch_matr)

        # fill the output.
        #for i, row in enumerate(rows):
        #    out[row] = sub_out[i]

        out = kernels.mm(patch_matr)

        # O x N x (h2*w2)
        out = out.view(kernels.size()[0], num_inputs, h2*w2)

        oc = out.size()[0]

        # N x O x (h2*w2)
        out = out.permute(1, 0, 2).contiguous()

        # N x O x h2 x w2
        out = out.view(num_inputs, oc, h2, w2)

        ctx.out_channels = out.size()[1]

        return out

    @staticmethod
    def backward(ctx, d_out):
        patch_matr, kernels, rows = ctx.saved_tensors
        H, W, in_channels, h1, w1, kernel_size, stride, padding = ctx.extra
        out_channels = ctx.out_channels

        active_kernels = ctx.intermediate

        num_inputs = d_out.size()[0]

        # N x O x (h2*w2)
        d_out = d_out.view(num_inputs, out_channels, H*W)

        # O x N x (h2*w2)
        d_out = d_out.permute(1, 0, 2).contiguous()

        # O x (N*h2*w2)
        d_out = d_out.view(out_channels, num_inputs*H*W)


        d_out_active_rows = d_out.index_select(0, rows)

        if d_out_active_rows.size() == torch.Size([0]):
            d_out_active_rows = d_out

        d_patch_matr = active_kernels.transpose(0,1).mm(d_out_active_rows)

        d_active_kernels = d_out_active_rows.mm(patch_matr.transpose(0,1))

        d_kernels = torch.zeros(kernels.size()).cuda()

        if rows.size() == torch.Size([0]):
            d_kernels = d_active_kernels
        else:
            d_kernels[rows] = d_active_kernels

        d_patches = d_patch_matr.view(ctx.patch_size)
        d_patches.permute(3,0,1,2,4,5).contiguous()

        d_x = P.col2im(d_patches, kernel_size, stride, padding)

        return d_x, d_kernels, None, None, None, None, None, None, \
               None, None, None





