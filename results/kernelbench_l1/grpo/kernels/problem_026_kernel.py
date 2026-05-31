import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = """#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Custom CUDA kernel for GELU activation using modular device functions
__device__ float gelu_func(const float* x, int idx) {
    float x_val = x[idx];
    float c = sqrtf(2.0f / M_PI);
    return 0.5f * x_val * (1.0f + tanhf(c * (x_val + 0.044715f * x_val * x_val * x_val)));
}

__global__ void gelu_kernel_modular(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = gelu_func(input, idx);
    }
}

torch::Tensor gelu_cuda_modular(torch::Tensor x) {
    auto size = x.numel();
    auto output = torch::empty_like(x);
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;
    gelu_kernel_modular<<<num_blocks, block_size>>>(x.data_ptr<float>(), output.data_ptr<float>(), size);
    return output;
}"""

cpp_source = """torch::Tensor gelu_cuda_modular(torch::Tensor x);"""

custom_ops = load_inline(
    name="custom_ops",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['gelu_cuda_modular'],
    verbose=False,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"],
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x):
        return custom_ops.gelu_cuda_modular(x)
