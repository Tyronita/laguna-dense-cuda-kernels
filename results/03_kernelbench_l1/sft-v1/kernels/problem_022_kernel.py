import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = """#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// CUDA kernel for Tanh activation
__global__ void tanh_kernel(const float* __restrict__ input, float* __restrict__ output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = tanhf(input[idx]);
    }
}

// PyTorch wrapper function
torch::Tensor tanh_cuda(torch::Tensor input) {
    auto size = input.numel();
    auto output = torch::empty_like(input);
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;
    tanh_kernel<<<num_blocks, block_size>>>(input.data_ptr<float>(), output.data_ptr<float>(), size);
    return output;
}

// PyTorch module
class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.tanh = tanh_cuda

    def forward(self, x):
        return self.tanh.forward(x)

// PyTorch extension"""

cpp_source = """torch::Tensor tanh_cuda(torch::Tensor input);"""

custom_ops = load_inline(
    name="custom_ops",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['tanh_cuda'],
    verbose=False,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"],
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x):
        return custom_ops.tanh_cuda(x)
