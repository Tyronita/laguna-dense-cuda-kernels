import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = """#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Optimized triangular matmul kernel using shared memory tiling and loop unrolling
__global__ void triangular_matmul_kernel(const float* __restrict__ A,
                                           const float* __restrict__ B,
                                           float* __restrict__ C,
                                           int N) {
    // Shared memory tile size
    const int TILE = 16;
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    float sum = 0.0f;

    // Loop over tiles
    for (int t = 0; t < (N + TILE - 1) / TILE; t++) {
        // Load A tile
        int a_col = t * TILE + threadIdx.x;
        if (row < N && a_col < N && a_col <= row)
            As[threadIdx.y][threadIdx.x] = A[row * N + a_col];
        else
            As[threadIdx.y][threadIdx.x] = 0.0f;

        // Load B tile
        int b_row = t * TILE + threadIdx.y;
        if (b_row < N && col < N && b_row >= col)
            Bs[threadIdx.y][threadIdx.x] = B[b_row * N + col];
        else
            Bs[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        // Compute partial sum for this tile
        #pragma unroll
        for (int k = 0; k < TILE; k++) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        __syncthreads();
    }

    // Write result
    if (row < N && col < N && row >= col)
        C[row * N + col] = sum;
}

// C++ interface exposed via PyBind11
torch::Tensor triangular_matmul_cuda(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "Input A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "Input B must be a CUDA tensor");
    TORCH_CHECK(A.dim() == 2, "Input A must be 2D");
    TORCH_CHECK(B.dim() == 2, "Input B must be 2D");
    TORCH_CHECK(A.size(0) == A.size(1), "Input A must be square");
    TORCH_CHECK(B.size(0) == B.size(1), "Input B must be square");
    TORCH_CHECK(A.size(0) == B.size(0), "Input A and B must be the same size");

    int N = A.size(0);
    auto C = torch::empty_like(A);

    const int TILE = 16;
    dim3 grid((N + TILE - 1) / TILE, (N + TILE - 1) / TILE);
    dim3 block(TILE, TILE);

    triangular_matmul_kernel<<<grid, block>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        N
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "Kernel failed: ", cudaGetErrorString(err));

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
