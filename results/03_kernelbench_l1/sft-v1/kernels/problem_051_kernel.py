import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = """#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void argmax_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int64_t size,
    const int64_t dim) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        int d = idx % dim;
        int base = idx / dim;
        float max_val = -FLT_MAX;
        for (int i = 0; i < dim; ++i) {
            float val = input[base * dim + d * i];
            max_val = fmaxf(max_val, val);
        }
        output[idx] = max_val;
    }
}

torch::Tensor argmax_cuda(torch::Tensor input) {
    auto size = input.numel();
    auto dim = input.dim();
    auto output = torch::empty_like(input);
    
    const int threads = 256;
    const int blocks = (size + threads - 1) / threads;
    
    argmax_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        size,
        dim
    );
    
    return output;
}"""

cpp_source = """torch::Tensor argmax_cuda(torch::Tensor input);"""

custom_ops = load_inline(
    name="custom_ops",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['argmax_cuda'],
    verbose=False,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"],
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x):
        return custom_ops.argmax_cuda(x)
