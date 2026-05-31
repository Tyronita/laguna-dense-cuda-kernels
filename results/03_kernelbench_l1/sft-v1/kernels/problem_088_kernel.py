import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = """#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void gelu_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int64_t size) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = input[idx];
        float c = sqrtf(2.0f / M_PI);
        output[idx] = 0.5f * x * (1.0f + tanhf(c * (x + 0.044715f * x * x)));
    }
}

torch::Tensor gelu_cuda(torch::Tensor input) {
    auto size = input.numel();
    auto output = torch::empty_like(input);
    const int threads = 256;
    const int blocks = (size + threads - 1) / threads;
    gelu_kernel<<<blocks, threads>>>(input.data_ptr<float>(), output.data_ptr<float>(), size);
    return output;
}"""

cpp_source = """torch::Tensor gelu_cuda(torch::Tensor input);"""

custom_ops = load_inline(
    name="custom_ops",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['gelu_cuda'],
    verbose=False,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"],
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x):
        return custom_ops.gelu_cuda(x)
