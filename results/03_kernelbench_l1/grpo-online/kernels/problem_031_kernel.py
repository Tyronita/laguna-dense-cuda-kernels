import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = """#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Custom CUDA kernel for elementwise addition
__global__ void elementwise_add_kernel(const float* a, const float* b, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = a[idx] + b[idx];
    }
}

// Custom CUDA kernel for ELU activation
__global__ void elu_kernel(const float* x, float* out, float alpha, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x_val = x[idx];
        out[idx] = fmaxf(x_val, 0.0f) * (1.0f + tanhf(x_val * alpha));
    }
}

// Custom CUDA function for elementwise addition
torch::Tensor elementwise_add_cuda(torch::Tensor a, torch::Tensor b) {
    auto size = a.numel();
    auto out = torch::zeros_like(a);
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;
    elementwise_add_kernel<<<num_blocks, block_size>>>(a.data_ptr<float>(), b.data_ptr<float>(), out.data_ptr<float>(), size);
    return out;
}

// Custom CUDA function for ELU activation
torch::Tensor elu_cuda(torch::Tensor x, float alpha) {
    auto size = x.numel();
    auto out = torch::zeros_like(x);
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;
    elu_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), out.data_ptr<float>(), alpha, size);
    return out;
}

// Pybind11 module definition"""

cpp_source = """torch::Tensor elementwise_add_cuda(torch::Tensor a, torch::Tensor b);\ntorch::Tensor elu_cuda(torch::Tensor x, float alpha);"""

custom_ops = load_inline(
    name="custom_ops",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['elementwise_add_cuda', 'elu_cuda'],
    verbose=False,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"],
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x):
        return custom_ops.elementwise_add_cuda(x)
