import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.cpp_extension import load_inline

gelu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void gelu_kernel(const float* x, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float xi = x[idx];
        float sqrt_2_over_pi = 0.7978845608f;
        float val = xi + 0.044715f * xi * xi * xi;
        out[idx] = 0.5f * xi * (1.0f + tanf(sqrt_2_over_pi * val));
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
"""
gelu_cpp_source = "torch::Tensor gelu_cuda(torch::Tensor x);"
gelu = load_inline(name="gelu", cpp_sources=gelu_cpp_source, cuda_sources=gelu_source, functions=["gelu_cuda"], verbose=True)

class ModelNew(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    def __init__(self):
        super(ModelNew, self).__init__()
        self.gelu = gelu

    def forward(self, x):
        return self.gelu.gelu_cuda(x)