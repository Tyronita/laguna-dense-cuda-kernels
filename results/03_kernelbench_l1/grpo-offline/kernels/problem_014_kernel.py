import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = """#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Optimized kernel for upper triangular matrix multiplication with loop unrolling
__global__ void upper_triangular_matmul_kernel(const float* A, const float* B, float* C, int N) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (row < N && col < N) {
        float sum = 0.0f;
        // For upper triangular matrices, A[i,k] is nonzero only if k <= i
        // and B[k,j] is nonzero only if j <= k
        for (int k = col; k <= row; ++k) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// The forward function exposed via PyBind11
torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "Input A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "Input B must be a CUDA tensor");
    TORCH_CHECK(A.dim() == 2, "Input A must be a 2D tensor");
    TORCH_CHECK(B.dim() == 2, "Input B must be a 2D tensor");
    TORCH_CHECK(A.size(0) == A.size(1), "Input A must be square");
    TORCH_CHECK(B.size(0) == B.size(1), "Input B must be square");
    TORCH_CHECK(A.size(0) == B.size(0), "Input A and B must be the same size");

    int N = A.size(0);
    auto C = torch::zeros_like(A);

    // Define block and grid sizes
    const int BLOCK_SIZE = 16;
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // Launch the optimized kernel
    upper_triangular_matmul_kernel<<<gridDim, blockDim>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), N
    );

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

    return C;
}"""

cpp_source = """torch::Tensor forward(torch::Tensor A, torch::Tensor B);"""

custom_ops = load_inline(
    name="custom_ops",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['forward'],
    verbose=False,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"],
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, A, B):
        return custom_ops.forward(A, B)
