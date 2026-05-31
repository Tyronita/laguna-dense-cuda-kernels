import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = """#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Custom CUDA kernel for elementwise addition with loop unrolling
__global__ void elementwise_add_kernel(const float* a, const float* b, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = a[idx] + b[idx];
    }
}

// Custom CUDA kernel for elementwise multiplication with loop unrolling
__global__ void elementwise_mul_kernel(const float* a, const float* b, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = a[idx] * b[idx];
    }
}

// Custom CUDA kernel for elementwise activation with loop unrolling
__global__ void elementwise_activation_kernel(const float* a, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = a[idx] * (1.0f + tanhf(out[idx] + 0.044715f * a[idx] * a[idx]));
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

// Custom CUDA function for elementwise multiplication
torch::Tensor elementwise_mul_cuda(torch::Tensor a, torch::Tensor b) {
    auto size = a.numel();
    auto out = torch::zeros_like(a);
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;
    elementwise_mul_kernel<<<num_blocks, block_size>>>(a.data_ptr<float>(), b.data_ptr<float>(), out.data_ptr<float>(), size);
    return out;
}

// Custom CUDA function for elementwise activation
torch::Tensor elementwise_activation_cuda(torch::Tensor a) {
    auto size = a.numel();
    auto out = torch::zeros_like(a);
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;
    elementwise_activation_kernel<<<num_blocks, block_size>>>(a.data_ptr<float>(), out.data_ptr<float>(), size);
    return out;
}

// Custom CUDA function for fused elementwise addition and multiplication
torch::Tensor fused_elementwise_add_mul_cuda(torch::Tensor a, torch::Tensor b) {
    auto size = a.numel();
    auto out = torch::zeros_like(a);
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;
    elementwise_mul_kernel<<<num_blocks, block_size>>>(a.data_ptr<float>(), b.data_ptr<float>(), out.data_ptr<float>(), size);
    return out;
}

// Custom CUDA function for fused elementwise addition and activation
torch::Tensor fused_elementwise_add_activation_cuda(torch::Tensor a, torch::Tensor b) {
    auto size = a.numel();
    auto out = torch::zeros_like(a);
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;
    elementwise_add_kernel<<<num_blocks, block_size>>>(a.data_ptr<float>(), b.data_ptr<float>(), out.data_ptr<float>(), size);
    elementwise_activation_kernel<<<num_blocks, block_size>>>(out.data_ptr<float>(), out.data_ptr<float>(), size);
    return out;
}

// Optimized Swish activation using fused elementwise addition and activation
torch::Tensor optimized_swish_cuda(torch::Tensor x) {
    return elementwise_activation_cuda(x);
}

// Optimized fused elementwise addition and multiplication
torch::Tensor optimized_fused_add_mul_cuda(torch::Tensor a, torch::Tensor b) {
    return fused_elementwise_add_mul_cuda(a, b);
}

// Optimized fused elementwise addition and activation
torch::Tensor optimized_fused_add_activation_cuda(torch::Tensor a, torch::Tensor b) {
    return fused_elementwise_add_activation_cuda(a, b);
}

// Optimized model with fused operations
torch::Tensor optimized_model_forward(torch::Tensor a, torch::Tensor b) {
    return optimized_fused_add_activation_cuda(a, b);
}"""

cpp_source = """torch::Tensor elementwise_add_cuda(torch::Tensor a, torch::Tensor b);\ntorch::Tensor elementwise_mul_cuda(torch::Tensor a, torch::Tensor b);\ntorch::Tensor elementwise_activation_cuda(torch::Tensor a);\ntorch::Tensor fused_elementwise_add_mul_cuda(torch::Tensor a, torch::Tensor b);\ntorch::Tensor fused_elementwise_add_activation_cuda(torch::Tensor a, torch::Tensor b);\ntorch::Tensor optimized_swish_cuda(torch::Tensor x);\ntorch::Tensor optimized_fused_add_mul_cuda(torch::Tensor a, torch::Tensor b);\ntorch::Tensor optimized_fused_add_activation_cuda(torch::Tensor a, torch::Tensor b);\ntorch::Tensor optimized_model_forward(torch::Tensor a, torch::Tensor b);"""

custom_ops = load_inline(
    name="custom_ops",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['elementwise_add_cuda', 'elementwise_mul_cuda', 'elementwise_activation_cuda', 'fused_elementwise_add_mul_cuda', 'fused_elementwise_add_activation_cuda', 'optimized_swish_cuda', 'optimized_fused_add_mul_cuda', 'optimized_fused_add_activation_cuda', 'optimized_model_forward'],
    verbose=False,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"],
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x):
        return custom_ops.elementwise_add_cuda(x)
