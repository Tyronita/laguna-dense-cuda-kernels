import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = """#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void gelu_kernel(const float* x, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x_val = x[idx];
        float c = 0.5 * x_val * (1.0 + tanhf(sqrt(2.0 / pi) * (x_val + 0.044715 * x_val * x_val)));
        out[idx] = c;
    }
}

torch::Tensor gelu_cuda(torch::Tensor x) {
    auto size = x.numel();
    auto out = torch::zeros_like(x);
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;
    gelu_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), out.data_ptr<float>(), size);
    return out;
}

torch::Tensor elementwise_add_cuda(torch::Tensor a, torch::Tensor b) {
    auto size = a.numel();
    auto out = torch::zeros_like(a);
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;
    elementwise_add_kernel<<<num_blocks, block_size>>>(a.data_ptr<float>(), b.data_ptr<float>(), out.data_ptr<float>(), size);
    return out;
}

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.elementwise_add = elementwise_add_cuda
        self.gelu = gelu_cuda

    def forward(self, a, b):
        return self.gelu(a + self.elementwise_add(a, b))"""

cpp_source = """torch::Tensor gelu_cuda(torch::Tensor x);\ntorch::Tensor elementwise_add_cuda(torch::Tensor a, torch::Tensor b);"""

custom_ops = load_inline(
    name="custom_ops",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['gelu_cuda', 'elementwise_add_cuda'],
    verbose=False,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"],
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x):
        return custom_ops.gelu_cuda(x)
