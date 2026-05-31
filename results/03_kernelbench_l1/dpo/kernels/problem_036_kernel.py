import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = """#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void elementwise_add_kernel(const float* a, const float* b, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) { out[idx] = a[idx] + b[idx];
    }
}

torch::Tensor elementwise_add_cuda(torch::Tensor a, torch::Tensor b) {
    auto size = a.numel();
    auto out = torch::zeros_like(a);
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;
    elementwise_add_kernel<<<num_blocks, block_size>>>(a.data_ptr<float>(), b.data_ptr<float>(), out.data_ptr<float>(), size);
    return out;
}

torch::Tensor rms_norm_cuda(torch::Tensor x, int64_t num_features, float eps) {
    auto size = x.numel();
    auto out = torch::empty_like(x);
    
    // Calculate mean and variance over the feature dimension
    auto mean = torch::mean(x.view({-1, num_features, -1, -1}), 1, true);
    auto variance = torch::pow(x.view({-1, num_features, -1, -1}), 2, 1).sum(1, true);
    
    // Compute RMS normalization
    auto rms = torch::sqrt(variance + eps);
    
    // Apply elementwise division using the custom CUDA kernel
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;
    elementwise_add_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), rms.data_ptr<float>(), out.data_ptr<float>(), size);
    
    return out;
}

torch::Tensor forward(torch::Tensor x, int64_t num_features, float eps) {
    return rms_norm_cuda(x, num_features, eps);
}"""

cpp_source = """torch::Tensor elementwise_add_cuda(torch::Tensor a, torch::Tensor b);\ntorch::Tensor rms_norm_cuda(torch::Tensor x, int64_t num_features, float eps);\ntorch::Tensor forward(torch::Tensor x, int64_t num_features, float eps);"""

custom_ops = load_inline(
    name="custom_ops",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['elementwise_add_cuda', 'rms_norm_cuda', 'forward'],
    verbose=False,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"],
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x):
        return custom_ops.elementwise_add_cuda(x)
