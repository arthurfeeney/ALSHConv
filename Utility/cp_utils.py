
import torch
import cupy
import cupy.cuda as function
from string import Template
from collections import namedtuple


CUDA_NUM_THREADS = 1024

def GET_BLOCKS(N, NUM_THREADS=None):
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

    int this_thread = blockIdx.x * blockDim.x + threadIdx.x;

    CUDA_KERNEL_LOOP(index, ${n}) {
        dst[(this_thread * num_buckets) + (votes[index] % num_buckets)]++;
    }
}
'''

def count_votes(votes, table_size, device=torch.device('cuda')):
    r"""
    counts up the votes for each bucket in the table.
    Uses _count_votes_kernel if it is on device is cuda
    returns tallies for each hash function.
    """
    num_hashes = votes.size(0)
    num_votes_per_hash = votes.size(1)

    tallies_dim = torch.Size([num_hashes, table_size])

    if device == torch.device('cpu'):
        tallies = torch.empty(tallies_dim).long().to(device).fill_(0)
        for h in range(num_hashes):
            for vote in votes[h]:
                tallies[h][vote.long() % table_size] += 1
        return tallies

    r"""
    non-optimal. Uses separate kernel for each hash. 
    """
    tallies = torch.empty(tallies_dim).long().cuda()
    for h in range(num_hashes):
        with torch.cuda.device_of(votes):
            v_per_thread = torch.empty(CUDA_NUM_THREADS, 
                                       table_size).cuda().long().fill_(0)
            
            f = load_kernel('counts', _count_votes_kernel, 
                            n=num_votes_per_hash, table_size=table_size)
            f(block=(CUDA_NUM_THREADS, 1, 1),
              grid=(1, 1, 1),
              args=[v_per_thread.data_ptr(), votes[h].data_ptr()],
              stream=Stream(ptr=torch.cuda.current_stream().cuda_stream))

            tallies[h] = v_per_thread.sum(dim=0)
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
            dst[f] = src[index]*h*w+i;
        }
    }
}
'''

def get_true_las(las, kernel_size):
    n = las.size(0)

    true_las = torch.empty(n*kernel_size**2).long().cuda()

    with torch.cuda.device_of(las):
        f = load_kernel('true_las', _true_las_kernel, n=n,
                        kernel_size=kernel_size)
        f(block=(1, 1, 1),
          grid=(1, 1, 1),
          args=[true_las.data_ptr(), las.data_ptr()],
          stream=Stream(ptr=torch.cuda.current_stream().cuda_stream))

    return true_las


_rehash_kernel = '''

#define CUDA_KERNEL_LOOP(i, n)                          \
    for(int i = blockIdx.x * blockDim.x + threadIdx.x;  \
        i < (n);                                        \
        i += blockDim.x * gridDim.x)

extern "C"
__global__ void rehash(long long* table, long long* table_row_lengths,
                       long long* indices, long long* rows) {

    int bucket_len = ${num_kernels};

    CUDA_KERNEL_LOOP(idx, ${n}) {
        long long index = indices[idx];
        long long bucket = bucket_len * index;
        table[bucket + table_row_lengths[index]] = rows[idx];
        table_row_lengths[index]++;
    }
}
'''

def rehash_alsh_table(table, table_row_lengths, indices, rows, table_size,
                      out_channels):
    r"""
    This function should modify table in place
    """

    n = rows.size(0)

    with torch.cuda.device_of(table):
        f = load_kernel('rehash', _rehash_kernel, n=n,
                        num_kernels=out_channels)
        f(block=(1, 1, 1),
          grid=(1, 1, 1),
          args=[table.data_ptr(), table_row_lengths.data_ptr(),
                indices.data_ptr(), rows.data_ptr()],
          stream=Stream(ptr=torch.cuda.current_stream().cuda_stream))

    return table, table_row_lengths


_unique_kernel = '''
#define CUDA_KERNEL_LOOP(i, n)                          \
    for(int i = blockIdx.x * blockDim.x + threadIdx.x;  \
        i < (n);                                        \
        i += blockDim.x * gridDim.x)

extern "C"
__global__ void unique(${Dtype}* dst, unsigned char* dst_indices, 
                       ${Dtype}* src) 
{
    CUDA_KERNEL_LOOP(idx, ${n}) {
        
        int i = 0;
        while(dst[i] != src[idx] && i < ${n}) {
            ++i;
        }

        if(i == ${n}) {
            dst[idx] = src[idx];
            dst_indices[idx] = 1;
        }
    }
}
'''

def get_unique(input, sorted=False, device=torch.device('cuda')):
    r"""
    takes a tensor and device.
    returns all the unique elements input. 
    same as torch.unique().
    """
    if device == torch.device('cpu'):
        return input.unique(sorted=sorted)

    # GPU version is extremely inneficient. Just keeps tensor on device 

    n = input.size(0)
    Dtype = type_string(input)

    dst = torch.empty(n).to(input).fill_(0)
    dst_indices = torch.empty(n).cuda().byte().fill_(0)

    zero_indices = (dst == 0).nonzero()

    if zero_indices.numel() > 0:
        dst_indices[zero_indices[0]] = 1

    with torch.cuda.device_of(input):
        f = load_kernel('unique', _unique_kernel, n=n, 
                        Dtype=Dtype)
        f(block=(1, 1, 1),
          grid=(1, 1, 1),
          args=[dst.data_ptr(), dst_indices.data_ptr(), input.data_ptr()],
          stream=Stream(ptr=torch.cuda.current_stream().cuda_stream))
    
    return dst[dst_indices]


def fill_powers_of_norms(input, m, sub, device=torch.device('cuda')):
    if input.dim() == 0:
        return fill_powers_of_norm(input, m, sub, device=device)
    elif input.dim() == 1:
        return fill_powers_of_norm_2d(input, m, sub, device=device)

_power_fill_kernel = '''
#define CUDA_KERNEL_LOOP(i, n)                          \
    for(int i = blockIdx.x * blockDim.x + threadIdx.x;  \
        i < (n);                                        \
        i += blockDim.x * gridDim.x)

extern "C"
__global__ void power_fill_2d(float* dst, float* norms)
{
    // n is the number of norms. 
    // m is the number of cols / append length.
    
    int append_len = ${m};

    CUDA_KERNEL_LOOP(idx, ${n}) {
        for (int col = blockIdx.y * blockDim.y + threadIdx.y; 
             col < append_len; 
             col += blockDim.y * gridDim.y) 
        {
            float exp = powf(2.0f, (float)col+1);
            if(exp > 1000) {
                dst[idx * append_len + col] = 0.5f; 
            }
            else {
                dst[idx * append_len + col] = 0.5f - powf(norms[idx], exp);
            }
        }
    }
}

extern "C"
__global__ void no_sub_power_fill_2d(float* dst, float* norms)
{
    // n is the number of norms. 
    // m is the number of cols / append length.
    
    int append_len = ${m};

    CUDA_KERNEL_LOOP(idx, ${n}) {
        for (int col = blockIdx.y * blockDim.y + threadIdx.y; 
             col < append_len; 
             col += blockDim.y * gridDim.y) 
        {
            float exp = powf(2.0f, (float)col+1);
            if(exp > 1000) {
                dst[idx * append_len + col] = 0.0f; 
            }
            else {
                dst[idx * append_len + col] = powf(norms[idx], exp);
            }
        }
    }
}
'''

def fill_powers_of_norm_2d(input, m, sub, device=torch.device('cuda')):
    if device == torch.device('cpu'):
        powers = torch.empty(input.size(0), m).to(device)
        for i in range(m):
            for j, x_n in enumerate(input):
                exp = 2**(i+1)
                if sub: 
                    if exp < 1000:
                        powers[j, i] = .5 - x_n**exp
                    else:
                        powers[j,i] = .5
                else:
                    if exp < 1000:
                        powers[j, i] = x_n**exp
                    else:
                        powers[j,i] = 0.0
        return powers

    with torch.cuda.device_of(input):
        powers = torch.empty(input.size(0), m).to(device)
        if sub:
            f = load_kernel('power_fill_2d', _power_fill_kernel, m=m, 
                            n=input.size(0))
        else:
            f = load_kernel('no_sub_power_fill_2d', _power_fill_kernel, m=m, 
                            n=input.size(0))
        f(block=(CUDA_NUM_THREADS // m, m, 1),
          grid=(1, 1, 1),
          args=[powers.data_ptr(), input.data_ptr()],
          stream=Stream(ptr=torch.cuda.current_stream().cuda_stream))
        
        return powers

_power_fill_1d_kernel = '''
#define CUDA_KERNEL_LOOP(i, n)                          \
    for(int i = blockIdx.x * blockDim.x + threadIdx.x;  \
        i < (n);                                        \
        i += blockDim.x * gridDim.x)


extern "C"
__global__ void power_fill_1d(float* dst, float* norm) 
{
    CUDA_KERNEL_LOOP(idx, ${m}) {
        float exp = powf(2.0, (float)idx+1);
        
        if(exp > 1000) {
            dst[idx] = 0.5f;
        }
        else {
            dst[idx] = 0.5 - powf(*norm, exp);
        }
    }
}

extern "C"
__global__ void no_sub_power_fill_1d(float* dst, float* norm) 
{
    CUDA_KERNEL_LOOP(idx, ${m}) {
        float exp = powf(2.0, (float)idx+1);
        
        if(exp > 1000) {
            dst[idx] = 0.0f;
        }
        else {
            dst[idx] = powf(*norm, exp);
        }
    }
}
'''

def fill_powers_of_norm(input, m, sub, device=torch.device('cuda')):
    if device == torch.device('cpu'):
        if sub:
            powers = \
                torch.Tensor([.5 - input**2**(i+1) for i in range(m)])    
        else:
            powers = torch.Tensor([input**2**(i+1) for i in range(m)])    
        return power.to(device)
 
    with torch.cuda.device_of(input):
        powers = torch.empty(m).to(device)
        
        if sub:
            f = load_kernel('power_fill_1d', _power_fill_1d_kernel, m=m)
        else:
            f = load_kernel('no_sub_power_fill_1d', _power_fill_1d_kernel, 
                            m=m)

        f(block=(CUDA_NUM_THREADS, 1, 1),
          grid=(1, 1, 1),
          args=[powers.data_ptr(), input.data_ptr()],
          stream=Stream(ptr=torch.cuda.current_stream().cuda_stream))

        return powers
         

@cupy.util.memoize(for_each_device=True)
def load_kernel(kernel_name, code, **kwargs):
    if kwargs is not None:
        code = Template(code).substitute(**kwargs)
        kernel_code = cupy.cuda.compile_with_cache(code)
        return kernel_code.get_function(kernel_name)

def type_string(t):
    if isinstance(t, torch.cuda.FloatTensor):
        return 'float'
    elif isinstance(t, torch.cuda.DoubleTensor):
        return 'double'
    elif isinstance(t, torch.cuda.LongTensor):
        return 'long long'
    elif isinstance(t, torch.cuda.IntTensor):
        return 'int'


def zero_fill_missing(x, i, dims, device=torch.device('cuda')):
    r"""
    fills channels that weren't computed with zeros.
    """
    if i is None:
        return x
    t = torch.empty(dims).to(device).fill_(0)
    t[:,i,:,:] = x[:,]
    return t
