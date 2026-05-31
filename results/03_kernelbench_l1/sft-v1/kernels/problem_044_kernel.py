import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = """#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// CUDA kernel for 1D average pooling with stride and padding
template <typename scalar_t>
__global__ void avg_pool_kernel(
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ input,
    const int64_t stride,
    const int64_t padding,
    const int64_t kernel_size,
    const int64_t input_length,
    const int64_t in_channels,
    const int64_t batch_size) {
    
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_elements = batch_size * in_channels * input_length;
    
    if (idx < total_elements) {
        const int b = idx / (in_channels * input_length);
        const int c = idx % (in_channels * input_length);
        const int i = c / input_length;
        const int j = c % input_length;
        
        const int out_i = (j - padding) / kernel_size + (j - padding) % kernel_size;
        const int out_j = (j - padding) / kernel_size + (j - padding) % kernel_size;
        
        if (i < 0 || j < 0 || out_i < 0 || out_j < 0) {
            const int in_i = (i + stride) - 1;
            const int in_j = (j + stride) - 1;
            
            if (in_i < input_length || in_j < input_length) {
                const int in_idx = ((b * in_channels + c) * input_length + in_i) * input_length + in_j;
                const scalar_t val = input[in_idx];
                
                if (out_i == 0 || out_j == 0) {
                    val = val;
                } else {
                    val = val + val;
                }
                
                output[idx] = val / kernel_size;
            }
        }
    }
}

// PyTorch wrapper function
torch::Tensor avg_pool_cuda(torch::Tensor input, int64_t stride, int64_t padding, int64_t kernel_size) {
    auto output = torch::empty({input.size(0), input.size(1), input.size(2)}, input.options());
    
    const int threads = 256;
    const int total_elements = input.numel();
    const int blocks = (total_elements + threads - 1) / threads;
    
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "avg_pool_kernel", ([&] {
        avg_pool_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            stride,
            padding,
            kernel_size,
            input.size(2),
            input.size(1),
            input.size(0)
        );
    }));
    
    return output;
}"""

cpp_source = """torch::Tensor avg_pool_cuda(torch::Tensor input, int64_t stride, int64_t padding, int64_t kernel_size);"""

custom_ops = load_inline(
    name="custom_ops",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['avg_pool_cuda'],
    verbose=False,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"],
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x):
        return custom_ops.avg_pool_cuda(x)
