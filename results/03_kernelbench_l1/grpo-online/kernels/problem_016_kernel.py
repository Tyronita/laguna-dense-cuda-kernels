import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = """#include <torch/extension.h>
#include <cuda_runtime.h>

// Optimized CUDA kernel for matrix multiplication with loop unrolling
__global__ void matmul_unroll_kernel(const float* A, const float* B, float* C, int K, int M, int N) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < M && col < N) {
        float sum = 0.0f;
        // Unroll loop by 4 to reduce loop overhead
        #pragma unroll 4
        for (int k = 0; k < K; k += 4) {
            sum += A[k * M + row] * B[k * N + col] +
                   A[(k + 1) * M + row] * B[(k + 1) * N + col] +
                   A[(k + 2) * M + row] * B[(k + 2) * N + col] +
                   A[(k + 3) * M + row] * B[(k + 3) * N + col];
        }
        C[row * N + col] = sum;
    }
}

// C++ interface exposed via PyBind11
torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.dim() == 2, "A must be 2D");
    TORCH_CHECK(B.dim() == 2, "B must be 2D");
    TORCH_CHECK(A.size(1) == B.size(0), "A and B must have same K dimension");
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Inputs must be on CUDA");
    TORCH_CHECK(A.is_contiguous() && B.is_contiguous(), "Inputs must be contiguous");

    int K = A.size(0);
    int M = A.size(1);
    TORCH_CHECK(B.size(1) == B.size(2), "B must have shape (K, N)");
    int N = B.size(1);

    auto C = torch::empty({M, N}, A.options());
    
    const int block_size = 16;
    dim3 grid((M + block_size - 1) / block_size, (N + block_size - 1) / block_size);
    dim3 block(block_size, block_size);
    
    matmul_unroll_kernel<<<grid, block>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), K, M, N
    );
    
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "Kernel failed: ", cudaGetErrorString(err));
    
    return C;
}"""

cpp_source = """torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B);"""

custom_ops = load_inline(
    name="custom_ops",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['matmul_cuda'],
    verbose=False,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"],
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, A, B):
        return custom_ops.matmul_cuda(A, B)
