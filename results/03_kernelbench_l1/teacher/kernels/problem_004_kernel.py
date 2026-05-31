import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

matmul_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void matmul_kernel(const float* A, const float* B, float* C, int M, int K) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k];
        }
        C[row] = sum;
    }
}

torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B) {
    int M = A.size(0);
    int K = A.size(1);
    auto C = torch::zeros({M}, torch::TensorOptions().dtype(A.dtype()).device(A.device()));
    const int block_size = 256;
    const int num_blocks = (M + block_size - 1) / block_size;
    matmul_kernel<<<num_blocks, block_size>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, K);
    return C;
}
"""
matmul_cpp_source = "torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B);"
matmul = load_inline(name="matmul", cpp_sources=matmul_cpp_source, cuda_sources=matmul_source, functions=["matmul_cuda"], verbose=True)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.matmul = matmul

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return self.matmul.matmul_cuda(A, B)