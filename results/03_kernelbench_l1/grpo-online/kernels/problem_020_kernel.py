import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = """#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>
#include <stdexcept>

// Custom CUDA kernel for elementwise addition with optimized block size
__global__ void optimized_block_size_elementwise_add_kernel(const float* a, const float* b, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = a[idx] + b[idx];
    }
}

// Custom CUDA kernel for LeakyReLU with optimized block size
__global__ void optimized_block_size_leaky_relu_kernel(const float* x, float* out, int size, float negative_slope) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = x[idx] > 0.0f ? x[idx] : x[idx] * negative_slope;
    }
}

// Custom CUDA function for elementwise addition
torch::Tensor elementwise_add_cuda(torch::Tensor a, torch::Tensor b) {
    auto size = a.numel();
    auto out = torch::zeros_like(a);
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;
    optimized_block_size_elementwise_add_kernel<<<num_blocks, block_size>>>(a.data_ptr<float>(), b.data_ptr<float>(), out.data_ptr<float>(), size);
    return out;
}

// Custom CUDA function for LeakyReLU
torch::Tensor leaky_relu_cuda(torch::Tensor x, float negative_slope) {
    auto size = x.numel();
    auto out = torch::zeros_like(x);
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;
    optimized_block_size_leaky_relu_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), out.data_ptr<float>(), size, negative_slope);
    return out;
}

// Pybind11 module definition"""

cpp_source = """torch::Tensor elementwise_add_cuda(torch::Tensor a, torch::Tensor b);\ntorch::Tensor leaky_relu_cuda(torch::Tensor x, float negative_slope);"""

custom_ops = load_inline(
    name="custom_ops",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['elementwise_add_cuda', 'leaky_relu_cuda'],
    verbose=False,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"],
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x):
        return custom_ops.elementwise_add_cuda(x)
