import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

matrix_scalar_mul_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void matrix_scalar_mul_kernel(const float* A, float s, float* C, int M, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = M * N;
    if (idx < total_elements) {
        int row = idx / N;
        int col = idx % N;
        C[idx] = A[idx] * s;
    }
}

torch::Tensor matrix_scalar_mul_cuda(torch::Tensor A, float s) {
    auto M = A.size(0);
    auto N = A.size(1);
    auto C = torch::zeros_like(A);
    const int block_size = 256;
    const int num_blocks = (M * N + block_size - 1) / block_size;
    matrix_scalar_mul_kernel<<<num_blocks, block_size>>>(A.data_ptr<float>(), s, C.data_ptr<float>(), M, N);
    return C;
}
"""
matrix_scalar_mul_cpp_source = "torch::Tensor matrix_scalar_mul_cuda(torch::Tensor A, float s);"
matrix_scalar_mul = load_inline(name="matrix_scalar_mul", cpp_sources=matrix_scalar_mul_cpp_source, cuda_sources=matrix_scalar_mul_source, functions=["matrix_scalar_mul_cuda"], verbose=True)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.matrix_scalar_mul = matrix_scalar_mul

    def forward(self, A: torch.Tensor, s: float) -> torch.Tensor:
        return self.matrix_scalar_mul.matrix_scalar_mul_cuda(A, s)