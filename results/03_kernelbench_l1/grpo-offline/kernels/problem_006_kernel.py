import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = """#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Define tile size for shared memory tiling
#define TILE_SIZE 32

// CUDA kernel for computing C = A * B using shared memory tiling
// A: shape (M, K), B: shape (K, N), C: shape (M, N)
__global__ void matmul_shared_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    // Compute the row and column indices for C
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;

    // Allocate shared memory for tiles of A and B
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    // Loop over tiles in the K dimension
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // Load a tile of A from global memory into shared memory
        int a_col = t * TILE_SIZE + threadIdx.x;
        if (row < M && a_col < K) {
            As[threadIdx.y][threadIdx.x] = A[row * K + a_col];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // Load a tile of B from global memory into shared memory
        int b_row = t * TILE_SIZE + threadIdx.y;
        if (b_row < K && col < N) {
            Bs[threadIdx.y][threadIdx.x] = B[b_row * N + col];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        // Compute partial dot product for this tile
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        __syncthreads();
    }

    // Write the result to C if within bounds
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// The forward function exposed via PyBind11
torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.dim() == 2, "A must be a 2D tensor");
    TORCH_CHECK(B.dim() == 2, "B must be a 2D tensor");
    TORCH_CHECK(A.size(1) == B.size(0), "A and B must have the same K dimension");
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Inputs must be on CUDA");
    TORCH_CHECK(A.is_contiguous() && B.is_contiguous(), "Inputs must be contiguous");

    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    auto C = torch::empty({M, N}, A.options());
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
    dim3 block(TILE_SIZE, TILE_SIZE);
    matmul_shared_kernel<<<grid, block>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, N, K);
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "Kernel failed: ", cudaGetErrorString(err));
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
