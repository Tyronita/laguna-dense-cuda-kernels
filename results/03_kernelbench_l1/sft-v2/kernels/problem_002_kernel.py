import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = """#include <torch/extension.h>
#include <cuda_runtime.h>

// Optimized CUDA kernel for matrix multiplication with loop unrolling and shared memory tiling
__global__ void matmul_kernel(const float* A, const float* B, float* C, int M, int K, int N) {
    // Shared memory tiles for A and B
    __shared__ float As[16][16];
    __shared__ float Bs[16][16];

    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    int row = by * 16 + ty;
    int col = bx * 16 + tx;

    float sum = 0.0f;

    // Loop over tiles
    for (int t = 0; t < (K + 15) / 16; ++t) {
        // Load A tile
        if (row < M && t * 16 + tx < K) {
            As[ty][tx] = A[row * K + t * 16 + tx];
        } else {
            As[ty][tx] = 0.0f;
        }

        // Load B tile
        if (t * 16 + ty < K && col < N) {
            Bs[ty][tx] = B[(t * 16 + ty) * N + col];
        } else {
            Bs[ty][tx] = 0.0f;
        }

        __syncthreads();

        // Compute partial sum with loop unrolling
        #pragma unroll
        for (int k = 0; k < 16; ++k) {
            sum += As[ty][k] * Bs[k][tx];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// C++ interface exposed to PyTorch
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

    // Define block and grid sizes
    const int TILE_SIZE = 16;
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);

    matmul_kernel<<<grid, block>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, K, N
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
