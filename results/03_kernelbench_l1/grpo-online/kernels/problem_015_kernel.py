import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = """#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Optimized triangular matmul kernel with loop unrolling and efficient thread mapping
__global__ void triangular_matmul_kernel(const float* A, const float* B, float* C, int N) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < N && col < N) {
        if (row < col) {
            C[row * N + col] = 0.0f;
        } else {
            float sum = 0.0f;
            // Unroll loop for better performance
            #pragma unroll 4
            for (int k = col; k <= row; ++k) {
                sum += A[row * N + k] * B[k * N + col];
            }
            C[row * N + col] = sum;
        }
    }
}

// C++ interface exposed via PyBind11
torch::Tensor triangular_matmul_cuda(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "Input A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "Input B must be a CUDA tensor");
    TORCH_CHECK(A.dim() == 2, "Input A must be a 2D tensor");
    TORCH_CHECK(B.dim() == 2, "Input B must be a 2D tensor");
    TORCH_CHECK(A.size(0) == A.size(1), "Input A must be square");
    TORCH_CHECK(B.size(0) == B.size(1), "Input B must be square");
    TORCH_CHECK(A.size(0) == B.size(0), "Input A and B must be the same size");

    int N = A.size(0);
    auto C = torch::zeros_like(A);

    // Use 16x16 block size for better occupancy
    const int block_size = 16;
    const int num_blocks = (N + block_size - 1) / block_size;

    triangular_matmul_kernel<<<num_blocks, block_size>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), N
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

    return C;
}"""

cpp_source = """torch::Tensor triangular_matmul_cuda(torch::Tensor A, torch::Tensor B);"""

custom_ops = load_inline(
    name="custom_ops",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['triangular_matmul_cuda'],
    verbose=False,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"],
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, A, B):
        return custom_ops.triangular_matmul_cuda(A, B)
