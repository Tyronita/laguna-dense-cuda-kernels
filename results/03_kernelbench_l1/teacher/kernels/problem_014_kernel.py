import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

matmul_triu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void matmul_triu_kernel(const float* A, const float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < N && col < N) {
        if (col >= row) {
            float sum = 0.0f;
            for (int k = 0; k < N; k++) {
                if (k >= row && k <= col) {
                    sum += A[row * N + k] * B[k * N + col];
                }
            }
            C[row * N + col] = sum;
        } else {
            C[row * N + col] = 0.0f;
        }
    }
}

torch::Tensor matmul_triu_cuda(torch::Tensor A, torch::Tensor B) {
    int N = A.size(0);
    auto C = torch::zeros({N, N}, A.options());
    
    const int block_size = 16;
    dim3 block(block_size, block_size);
    dim3 grid((N + block_size - 1) / block_size, (N + block_size - 1) / block_size);
    
    matmul_triu_kernel<<<grid, block>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), N);
    return C;
}
"""

matmul_triu_cpp_source = "torch::Tensor matmul_triu_cuda(torch::Tensor A, torch::Tensor B);"
matmul_triu = load_inline(
    name="matmul_triu",
    cpp_sources=matmul_triu_cpp_source,
    cuda_sources=matmul_triu_source,
    functions=["matmul_triu_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.matmul_triu = matmul_triu

    def forward(self, A, B):
        return self.matmul_triu.matmul_triu_cuda(A, B)