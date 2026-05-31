import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = """#include <torch/extension.h>
#include <cuda_runtime.h>

// Custom CUDA kernel for 3D tensor-matrix multiplication with loop unrolling
__global__ void matmul_unroll_kernel(const float* A, const float* B, float* C, int N, int M, int K, int L) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * M * L;
    if (idx < total) {
        int n = idx / (M * L);
        int m = (idx / L) % M;
        int l = idx % L;
        float sum = 0.0f;
        int k = 0;
        // Unroll loop by 4 for better performance
        #pragma unroll 4
        for (; k + 3 < K; k += 4) {
            sum += A[n * M * K + m * K + k] * B[k * L + l] +
                   A[n * M * K + m * K + k + 1] * B[(k + 1) * L + l] +
                   A[n * M * K + m * K + k + 2] * B[(k + 2) * L + l] +
                   A[n * M * K + m * K + k + 3] * B[(k + 3) * L + l];
        k += 4;
        }
        // Handle remaining elements
        for (; k < K; k++) {
            sum += A[n * M * K + m * K + k] * B[k * L + l];
        }
        C[n * M * L + m * L + l] = sum;
    }
}

// C++ interface exposed via PyBind11
torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.dim() == 3, "A must be 3D");
    TORCH_CHECK(B.dim() == 2, "B must be 2D");
    TORCH_CHECK(A.size(2) == B.size(0), "A.size(2) must match B.size(0)");
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Inputs must be on CUDA");
    TORCH_CHECK(A.is_contiguous() && B.is_contiguous(), "Inputs must be contiguous");

    int N = A.size(0), M = A.size(1), K = A.size(2);
    int L = B.size(1);
    auto C = torch::empty({N, M, L}, A.options());
    
    const int block_size = 256;
    const int num_blocks = (N * M * L + block_size - 1) / block_size;
    
    matmul_unroll_kernel<<<num_blocks, block_size>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), N, M, K, L
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
