import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

softsign_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void softsign_kernel(const float* x, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = x[idx] / (1.0f + fabsf(x[idx]));
    }
}

torch::Tensor softsign_cuda(torch::Tensor x) {
    auto size = x.numel();
    auto out = torch::zeros_like(x);
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;
    softsign_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), out.data_ptr<float>(), size);
    return out;
}
"""
softsign_cpp_source = "torch::Tensor softsign_cuda(torch::Tensor x);"
softsign = load_inline(name="softsign", cpp_sources=softsign_cpp_source, cuda_sources=softsign_source, functions=["softsign_cuda"], verbose=True)

class ModelNew(nn.Module):
    """
    Simple model that performs a Softsign activation.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
        self.softsign = softsign

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Softsign activation to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of any shape.

        Returns:
            torch.Tensor: Output tensor with Softsign applied, same shape as input.
        """
        return self.softsign.softsign_cuda(x)