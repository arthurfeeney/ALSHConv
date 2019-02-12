

from torch.utils.ffi import create_extension

ffi = create_extension(
    name='_ext.get_active_set',
    headers='get_active_set.h',
    sources=['get_active_set.c'],
    extra_compile_args=['-std=c99'],
    with_cuda=False)

ffi.build()
