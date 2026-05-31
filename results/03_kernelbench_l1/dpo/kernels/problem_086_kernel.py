import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = """#include <torch/extension.h>
#include <cuda_runtime.h>

// Custom CUDA kernel for elementwise addition with optimized block size
__global__ void elementwise_add_kernel(const float* a, const float* b, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = a[idx] + b[idx];
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

// Custom CUDA kernel for elementwise multiplication with optimized block size
__global__ void elementwise_mul_kernel(const float* a, const float* b, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = a[idx] * b[idx];
    }
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

// Custom CUDA kernel for elementwise ReLU with optimized block size
__global__ void elementwise_relu_kernel(const float* a, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = a[idx] > 0.0f ? a[idx] : 0.0f;
    }
}

// Custom CUDA function for elementwise ReLU
torch::Tensor elementwise_relu_cuda(torch::Tensor a) {
    auto size = a.numel();
    auto out = torch::zeros_like(a);
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;
    elementwise_relu_kernel<<<num_blocks, block_size>>>(a.data_ptr<float>(), out.data_ptr<float>(), size);
    return out;
}

// Custom CUDA kernel for elementwise addition with optimized block size
__global__ void elementwise_add_kernel(const float* a, const float* b, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = a[idx] + b[idx];
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

// Optimized depthwise convolution with custom CUDA kernels
torch::Tensor optimized_depthwise_conv2d(torch::Tensor x,
                                           torch::Tensor weight,
                                           torch::Tensor bias,
                                           int stride,
                                           int padding,
                                           int dilation) {
    // Use PyTorch's optimized conv2d for better performance
    return torch::conv2d(x, weight, bias, {stride, stride}, {padding, padding}, {dilation, dilation}, 1);
}

// Optimized pointwise convolution with custom CUDA kernels
torch::Tensor optimized_pointwise_conv2d(torch::Tensor x,
                                           torch::Tensor weight,
                                           torch::Tensor bias) {
    // Use PyTorch's optimized conv2d for better performance
    return torch::conv2d(x, weight, bias, {1, 1}, {0, 0}, {1, 1}, 1);
}

// Optimized module function with custom CUDA operators
torch::Tensor optimized_module_fn(torch::Tensor x,
                                torch::Tensor depthwise_weight,
                                torch::Tensor depthwise_bias,
                                torch::Tensor pointwise_weight,
                                torch::Tensor pointwise_bias,
                                int stride,
                                int padding,
                                int dilation) {
    // Optimized depthwise convolution using PyTorch's optimized conv2d
    x = optimized_depthwise_conv2d(x, depthwise_weight, depthwise_bias, stride, padding, dilation);
    
    // Optimized pointwise convolution using PyTorch's optimized conv2d
    x = optimized_pointwise_conv2d(x, pointwise_weight, pointwise_bias);
    
    return x;
}"""

cpp_source = """torch::Tensor elementwise_add_cuda(torch::Tensor a, torch::Tensor b);\ntorch::Tensor elementwise_mul_cuda(torch::Tensor a, torch::Tensor b);\ntorch::Tensor elementwise_relu_cuda(torch::Tensor a);\ntorch::Tensor elementwise_add_cuda(torch::Tensor a, torch::Tensor b);\ntorch::Tensor optimized_depthwise_conv2d(torch::Tensor x,
                                           torch::Tensor weight,
                                           torch::Tensor bias,
                                           int stride,
                                           int padding,
                                           int dilation);\ntorch::Tensor optimized_pointwise_conv2d(torch::Tensor x,
                                           torch::Tensor weight,
                                           torch::Tensor bias);\ntorch::Tensor optimized_module_fn(torch::Tensor x,
                                torch::Tensor depthwise_weight,
                                torch::Tensor depthwise_bias,
                                torch::Tensor pointwise_weight,
                                torch::Tensor pointwise_bias,
                                int stride,
                                int padding,
                                int dilation);"""

custom_ops = load_inline(
    name="custom_ops",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['elementwise_add_cuda', 'elementwise_mul_cuda', 'elementwise_relu_cuda', 'elementwise_add_cuda', 'optimized_depthwise_conv2d', 'optimized_pointwise_conv2d', 'optimized_module_fn'],
    verbose=False,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"],
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x):
        return custom_ops.elementwise_add_cuda(x)
