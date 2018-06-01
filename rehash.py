

import torch
import cupy
import cupy.cuda as function

from utils import load_kernel

_rehash_kernel = '''
#define CUDA_KERNEL_LOOP(i, n)                          \
    for(int i = blockId.x * blockDim.x + threadIdx.x;   \
        i < (n);                                        \
        i += blockDim.x * gridDim.x)

extern "C"
__global__ void rehash(long *table, long index, long* rows, float* weights)
{
    int i = blockId.x * blockDim.x + threadIdx.x;
    CUDA_KERNEL_LOOP(index, ${n}) { // n == len(rows)
        long row = rows[index];

        int row_len =

        int table_size = ${table_size};

        float* weight_row = malloc(sizeof(float) * );

    }

}
'''





