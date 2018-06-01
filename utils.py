
import torch
import cupy
import cupy.cuda as function
from string import Template
from collections import namedtuple

_flip_kernel = '''
extern "C"
__global__ void flip(float *dst, float *src, int w, int total) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i <= total)
        dst[i] = src[(i / w) * w + (w - (i % w) - 1)];
}
'''

Stream = namedtuple('Stream', ['ptr'])

@cupy.util.memoize(for_each_device=True)
def load_kernel(kernel_name, code, **kwargs):
    if kwargs is not None:
        code = Template(code).substitute(**kwargs)
        kernel_code = cupy.cuda.compile_with_cache(code)
        return kernel_code.get_function(kernel_name)
