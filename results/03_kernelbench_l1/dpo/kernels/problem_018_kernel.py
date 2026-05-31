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
            sum += A[(k + 0) * M + row] * B[col * K + (k + 0)]
                 + A[(k + 1) * M + row] * B[col * K + (k + 1)]
                 + A[(k + 2) * M + row] * B[col * K + (k + 2)]
                 + A[(k + 3) * M + row] * B[col * K + (k + 3)];
        }
        C[row * N + col] = sum;
    }
}

// C++ interface exposed to PyTorch
torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.dim() == 2, "A must be 2D");
    TORCH_CHECK(B.dim() == 2, "B must be 2D");
    TORCH_CHECK(A.size(1) == B.size(1), "A and B must have same K dimension");
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Inputs must be on CUDA");
    TORCH_CHECK(A.is_contiguous() && B.is_contiguous(), "Inputs must be contiguous");

    int K = A.size(1);
    int M = A.size(0);
    TORCH_CHECK(B.size(0) == K, "B must have same K dimension");
    int N = B.size(1);

    auto C = torch::zeros({M, N}, A.options());
    
    // Use 16x16 block size for better occupancy
    const int block_size = 16;
    dim3 grid((M + block_size - 1) / block_size, (N + block_size - 1) / block_size);
    
    matmul_unroll_kernel<<<grid, dim3(block_size, block_size)>>>(
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
