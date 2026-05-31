import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = """#include <torch/extension.h>
#include <cuda_runtime.h>

// Custom CUDA kernel for matrix multiplication with loop unrolling
__global__ void matmul_unroll_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < M && col < N) {
        float sum = 0.0f;
        // Unroll loop by 4 to reduce loop overhead
        #pragma unroll 4
        for (int k = 0; k < K; k += 4) {
            sum += A[row * K + k] * B[k * N + col] +
                   A[row * K + k + 1] * B[(k + 1) * N + col] +
                   A[row * K + k + 2] * B[(k + 2) * N + col] +
                   A[row * K + k + 3] * B[(k + 3) * N + col];
        }
        C[row * N + col] = sum;
    }
}

// C++ interface exposed via PyBind11
torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "Input A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "Input B must be a CUDA tensor");
    TORCH_CHECK(A.dtype() == torch::kFloat32, "Input A must be float32");
    TORCH_CHECK(B.dtype() == torch::kFloat32, "Input B must be float32");

    int M = A.size(0);
    int K = A.size(1);
    TORCH_CHECK(B.size(1) == K, "Dimension mismatch: A and B must have same K dimension");
    int N = B.size(0);

    auto C = torch::zeros({M, N}, A.options());

    // Use 16x16 block size for better occupancy
    const int block_size = 16;
    dim3 grid((M + block_size - 1) / block_size, (N + block_size - 1) / block_size);
    dim3 block(block_size, block_size);

    matmul_unroll_kernel<<<grid, block>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, N, K
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
