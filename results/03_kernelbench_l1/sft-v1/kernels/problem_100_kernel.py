import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = """#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void clamp_kernel(
    const float* __restrict__ input,
    const float* __restrict__ target,
    float* __restrict__ output,
    const int64_t size) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = (input[idx] - target[idx]) * 1.0f;
    }
}

torch::Tensor forward_cuda(torch::Tensor input, torch::Tensor target) {
    auto size = input.numel();
    auto output = torch::empty_like(input);
    
    const int threads = 256;
    const int blocks = (size + threads - 1) / threads;
    
    clamp_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        target.data_ptr<float>(),
        output.data_ptr<float>(),
        size
    );
    
    return output;
}"""

cpp_source = """torch::Tensor forward_cuda(torch::Tensor input, torch::Tensor target);"""

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

    def forward(self, predictions, targets):
        return custom_ops.forward_cuda(predictions, targets)
