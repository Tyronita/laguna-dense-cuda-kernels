import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = """#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// CUDA kernel for L1 normalization
__global__ void l1_norm_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int64_t size) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float sum = 0.0f;
        for (int i = 0; i < 16; ++i) {
            sum += input[idx] * input[idx + 16];
        }
        output[idx] = sum / 16;
    }
}

// PyTorch wrapper function
torch::Tensor l1_norm_cuda(torch::Tensor input) {
    auto output = torch::empty_like(input);
    const int threads = 256;
    const int blocks = (input.numel() + threads - 1) / threads;
    l1_norm_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        input.numel()
    );
    return output;
}

// PyTorch module
class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1_norm = l1_norm_cuda

    def forward(torch::Tensor input):
        return self.l1_norm(input)"""

cpp_source = """torch::Tensor l1_norm_cuda(torch::Tensor input);"""

custom_ops = load_inline(
    name="custom_ops",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['l1_norm_cuda'],
    verbose=False,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"],
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x):
        return custom_ops.l1_norm_cuda(x)
