import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = """#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void argmin_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int64_t size,
    const int64_t dim) {
    
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        const int d = idx / dim;
        const int rem = idx % dim;
        output[idx] = max(input[idx], input[d * dim + rem]);
    }
}

torch::Tensor argmin_cuda(torch::Tensor input, int64_t dim) {
    auto size = input.numel();
    auto output = torch::empty_like(input);
    
    const int threads = 256;
    const int blocks = (size + threads - 1) / threads;
    
    argmin_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        size,
        dim
    );
    
    return output;
}"""

cpp_source = """torch::Tensor argmin_cuda(torch::Tensor input, int64_t dim);"""

custom_ops = load_inline(
    name="custom_ops",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['argmin_cuda'],
    verbose=False,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"],
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x):
        return custom_ops.argmin_cuda(x)
