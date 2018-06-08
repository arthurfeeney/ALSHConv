
import torch
import cupy
import cupy.cuda as function
from string import Template
from collections import namedtuple


CUDA_NUM_THREADS = 1024

def GET_BLOCKS(N):
    return (N + CUDA_NUM_THREADS - 1) // CUDA_NUM_THREADS

Stream = namedtuple('Stream', ['ptr'])


_count_votes_kernel = '''
#define CUDA_KERNEL_LOOP(i, n)                          \
    for(int i = blockIdx.x * blockDim.x + threadIdx.x;  \
        i < (n);                                        \
        i += blockDim.x * gridDim.x)

extern "C"
__global__ void counts(long long* dst, long long* votes) {

    int num_buckets = ${table_size};

    CUDA_KERNEL_LOOP(index, ${n}) {
        ++dst[votes[index] % num_buckets];
    }
}
'''

def count_votes(votes, table_size):
    n = votes.size()[0]

    tallies = torch.empty(table_size).long().cuda().fill_(0)

    with torch.cuda.device_of(votes):
        f = load_kernel('counts', _count_votes_kernel, n=n,
                        table_size=table_size)
        f(block=(CUDA_NUM_THREADS, 1, 1),
          grid=(GET_BLOCKS(n), 1, 1),
          args=[tallies.data_ptr(), votes.data_ptr()],
          stream=Stream(ptr=torch.cuda.current_stream().cuda_stream))

    return tallies

_true_las_kernel = '''
#define CUDA_KERNEL_LOOP(i, n)                          \
    for(int i = blockIdx.x * blockDim.x + threadIdx.x;  \
        i < (n);                                        \
        i += blockDim.x * gridDim.x)

extern "C"
__global__ void true_las(long long* dst, long long* src) {

    long long h = ${kernel_size};
    long long w = ${kernel_size};

    CUDA_KERNEL_LOOP(index, ${n}) { // n is the size of src
        for(long long f = index*h*w, i=0; f < (index+1)*h*w; ++f, ++i) {
            dst[f] = src[index]+i;
        }
    }
}
'''

def get_true_las(las, kernel_size):
    n = las.size()[0]

    true_las = torch.empty(n*kernel_size**2).long().cuda()

    with torch.cuda.device_of(las):
        f = load_kernel('true_las', _true_las_kernel, n=n,
                        kernel_size=kernel_size)
        f(block=(CUDA_NUM_THREADS, 1, 1),
          grid=(GET_BLOCKS(n), 1, 1),
          args=[true_las.data_ptr(), las.data_ptr()],
          stream=Stream(ptr=torch.cuda.current_stream().cuda_stream))

    return true_las



@cupy.util.memoize(for_each_device=True)
def load_kernel(kernel_name, code, **kwargs):
    if kwargs is not None:
        code = Template(code).substitute(**kwargs)
        kernel_code = cupy.cuda.compile_with_cache(code)
        return kernel_code.get_function(kernel_name)


def zero_fill_missing(x, i, dims):
    r"""
    fills channels that weren't computed with zeros.
    """
    t = torch.empty(dims).cuda().fill_(0)
    t[:,i,:,:] = x[:,]
    return t
