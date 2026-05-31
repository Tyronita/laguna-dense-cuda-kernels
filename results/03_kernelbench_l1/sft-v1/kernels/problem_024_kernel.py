import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = """#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// CUDA kernel for LogSoftmax with loop unrolling
template <typename scalar_t>
__global__ void logsoftmax_kernel_unrolled(
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ input,
    const int64_t size) {
    
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        const scalar_t val = input[idx];
        const scalar_t max_val = val;
        const scalar_t sum_val = 0;
        
        // Unroll the loop for better performance
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const scalar_t val_i = val - max_val;
            sum_val += val_i;
        }
        
        // Compute the final result
        const scalar_t result = (sum_val - max_val) / 4;
        output[idx] = result;
    }
}

// PyTorch wrapper function
torch::Tensor forward_cuda(torch::Tensor input) {
    auto output = torch::empty_like(input);
    const int threads = 256;
    const int blocks = (input.numel() + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "logsoftmax_kernel_unrolled", ([&] {
        logsoftmax_kernel_unrolled<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            input.numel()
        );
    }));

    return output;
}"""

cpp_source = """torch::Tensor forward_cuda(torch::Tensor input);"""

custom_ops = load_inline(
    name="custom_ops",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['forward_cuda'],
    verbose=False,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"],
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x):
        return custom_ops.forward_cuda(x)
