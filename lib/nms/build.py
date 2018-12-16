import os
import torch
from torch.utils.ffi import create_extension

#ffi 自定义扩展c代码


sources = ['src/nms.c']
headers = ['src/nms.h']
defines = []
with_cuda = False

if torch.cuda.is_available():#如果cuda可用
    print('Including CUDA code.')
    sources += ['src/nms_cuda.c']#源文件位置
    headers += ['src/nms_cuda.h']#头文件位置
    defines += [('WITH_CUDA', None)]#
    with_cuda = True

this_file = os.path.dirname(os.path.realpath(__file__))
print(this_file)
extra_objects = ['src/cuda/nms_kernel.cu.o']
extra_objects = [os.path.join(this_file, fname) for fname in extra_objects]

ffi = create_extension(
    '_ext.nms',
    headers=headers,
    sources=sources,
    define_macros=defines,
    relative_to=__file__,
    with_cuda=with_cuda,
    extra_objects=extra_objects
)

if __name__ == '__main__':
    ffi.build()#使用build生成文件。cu
    #生成成功就可以从_ext中倒入c模块
