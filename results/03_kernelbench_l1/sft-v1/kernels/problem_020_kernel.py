import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = """#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void leaky_relu_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const float* __restrict__ bias,
    const int64_t size,
    const float negative_slope
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = input[idx];
        output[idx] = val > 0 ? val : val * negative_slope;
        if (bias[idx]) {
            output[idx] = val + bias[idx];
        }
    }
}

torch::Tensor leaky_relu_cuda(torch::Tensor input, float negative_slope, torch::Tensor bias) {
    auto size = input.numel();
    auto output = torch::empty_like(input);
    
    const int threads = 256;
    const int blocks = (size + threads - 1) / threads;
    
    leaky_relu_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        bias.data_ptr<float>(),
        size,
        negative_slope
    );
    
    return output;
}"""

cpp_source = """torch::Tensor leaky_relu_cuda(torch::Tensor input, float negative_slope, torch::Tensor bias);"""

custom_ops = load_inline(
    name="custom_ops",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['leaky_relu_cuda'],
    verbose=False,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"],
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x):
        return custom_ops.leaky_relu_cuda(x)
