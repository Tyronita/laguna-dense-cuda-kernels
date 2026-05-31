import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = """#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Custom CUDA kernel for diagonal matmul with optimized thread and block indexing
__global__ void diag_matmul_kernel(const float* A, const float* B, float* C, int N, int M) {
    // Calculate global thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    // Loop over elements with stride
    for (int i = idx; i < N * M; i += stride) {
        // Calculate row and column from linear index
        int row = i / M;
        int col = i % M;
        
        // Use diagonal element from A to multiply B's corresponding element
        C[i] = A[row] * B[i];
    }
}

// C++ interface exposed via PyBind11
torch::Tensor diag_matmul_cuda(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.dim() == 1, "A must be a 1D tensor");
    TORCH_CHECK(B.dim() == 2, "B must be a 2D tensor");
    TORCH_CHECK(A.size(1) == B.size(0), "A.size(1) must match B.size(0)");
    TORCH_CHECK(A.size(1) == B.size(1), "A.size(1) must match B.size(1)");
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Inputs must be on CUDA");
    TORCH_CHECK(A.is_contiguous() && B.is_contiguous(), "Inputs must be contiguous");

    int N = A.size(0);
    int M = B.size(1);
    auto C = torch::empty_like(B);

    // Optimize thread and block configuration
    const int threads = 256;
    const int blocks = (N * M + threads - 1) / threads;

    diag_matmul_kernel<<<blocks, threads>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        N,
        M
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "Kernel failed: ", cudaGetErrorString(err));

    return C;
}"""

cpp_source = """torch::Tensor diag_matmul_cuda(torch::Tensor A, torch::Tensor B);"""

custom_ops = load_inline(
    name="custom_ops",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['diag_matmul_cuda'],
    verbose=False,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"],
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, A, B):
        return custom_ops.diag_matmul_cuda(A, B)
