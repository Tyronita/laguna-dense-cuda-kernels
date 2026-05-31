import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = """#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// CUDA kernel for matrix multiplication with transposed B using shared memory tiling
template <typename scalar_t>
__global__ void matmul_transpose_shared_kernel(
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ A,
    const scalar_t* __restrict__ B,
    const int M,
    const int N,
    const int K) {
    
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    const int col = blockIdx.y * blockDim.y + threadIdx.y;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    
    scalar_t sum = 0;
    
    // Shared memory tiles for A and B
    __shared__ scalar_t As[16][16];
    __shared__ scalar_t Bs[16][16];
    
    // Loop over tiles
    for (int t = 0; t < (K + 15) / 16; ++t) {
        // Collaborative loading of A and B tiles into shared memory
        if (row < M && (t * 16 + ty) < K) {
            As[ty][tx] = A[row * K + t * 16 + ty];
        } else {
            As[ty][tx] = 0;
        }
        
        if (col < N && (t * 16 + tx) < K) {
            Bs[ty][tx] = B[col * K + t * 16 + tx];
        } else {
            Bs[ty][tx] = 0;
        }
        
        __syncthreads();
        
        // Compute partial dot product for this tile
        if (row < M && col < N) {
            #pragma unroll
            for (int k = 0; k < 16; ++k) {
                sum += As[k][tx] * Bs[ty][k];
            }
        }
        
        __syncthreads();
    }
    
    if (row < M && col < N) {
        output[row * N + col] = sum;
    }
}

// PyTorch wrapper function
torch::Tensor matmul_transpose_cuda(torch::Tensor A, torch::Tensor B) {
    const int M = A.size(0);
    const int K = A.size(1);
    const int N = B.size(0);

    auto options = torch::TensorOptions()
        .dtype(A.dtype())
        .device(A.device());
    auto output = torch::empty({M, N}, options);

    const int BLOCK_SIZE = 16;
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks((M + BLOCK_SIZE - 1) / BLOCK_SIZE,
                (N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    AT_DISPATCH_FLOATING_TYPES(A.type(), "matmul_transpose_shared_kernel", ([&] {
        matmul_transpose_shared_kernel<scalar_t><<<blocks, threads>>>(
            output.data_ptr<scalar_t>(),
            A.data_ptr<scalar_t>(),
            B.data_ptr<scalar_t>(),
            M, N, K
        );
    }));

    return output;
}"""

cpp_source = """torch::Tensor matmul_transpose_cuda(torch::Tensor A, torch::Tensor B);"""

custom_ops = load_inline(
    name="custom_ops",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['matmul_transpose_cuda'],
    verbose=False,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"],
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, A, B):
        return custom_ops.matmul_transpose_cuda(A, B)
