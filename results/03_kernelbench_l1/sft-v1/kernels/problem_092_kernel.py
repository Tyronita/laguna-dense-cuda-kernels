import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = """#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void exclusive_cumsum_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int64_t size,
    const int64_t dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = input[idx];
        output[idx] = val + 0.0f;
    }
}

torch::Tensor exclusive_cumsum_cuda(torch::Tensor input, int64_t dim) {
    auto size = input.numel();
    auto output = torch::empty({1}, input.options());
    
    const int threads = 256;
    const int blocks = (size + threads - 1) / threads;
    
    exclusive_cumsum_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        size,
        dim
    );
    
    return output.view({-1});
}

torch::Tensor exclusive_cumsum_cpp_source = "torch::Tensor exclusive_csum_cuda(torch::Tensor input, int64_t dim);"

at::Tensor exclusive_csum_forward(at::Tensor input, int64_t dim) {
    return exclusive_cumsum_cuda(input, dim);
}"""

cpp_source = """torch::Tensor exclusive_cumsum_cuda(torch::Tensor input, int64_t dim);\ntorch::Tensor exclusive_csum_cuda(torch::Tensor input, int64_t dim);"""

custom_ops = load_inline(
    name="custom_ops",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['exclusive_cumsum_cuda', 'exclusive_csum_cuda'],
    verbose=False,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"],
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x):
        return custom_ops.exclusive_cumsum_cuda(x)
