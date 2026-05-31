import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = """#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// CUDA kernel for matrix multiplication with shared memory tiling
template <typename scalar_t>
__global__ void matmul_shared_kernel(
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    const int M,
    const int N,
    const int K) {
    
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    const int col = blockIdx.y * blockDim.y + threadIdx.y;
    scalar_t sum = 0;
    
    // Shared memory tiles
    __shared__ scalar_t tileA[16][16];
    __shared__ scalar_t tileB[16][16];
    
    // Loop over tiles
    for (int t = 0; t < (K + 15) / 16; ++t) {
        // Collaborative loading of tiles into shared memory
        if (t * 16 + threadIdx.y < K && row < M) {
            tileA[threadIdx.y][threadIdx.x] = input[row * K + t * 16 + threadIdx.y];
        } else {
            tileA[threadIdx.y][threadIdx.x] = 0;
        }
        
        if (t * 16 + threadIdx.x < K && col < N) {
            tileB[threadIdx.y][threadIdx.x] = weight[col * K + t * 16 + threadIdx.x];
        } else {
            tileB[threadIdx.y][threadIdx.x] = 0;
        }
        
        __syncthreads();
        
        // Compute partial results for this tile
        if (row < M && col < N) {
            #pragma unroll
            for (int k = 0; k < 16; ++k) {
                sum += tileA[k][threadIdx.x] * tileB[threadIdx.y][k];
            }
        }
        
        __syncthreads();
    }
    
    if (row < M && col < N) {
        output[row * N + col] = sum;
    }
}

// PyTorch wrapper function
torch::Tensor matmul_cuda(torch::Tensor input, torch::Tensor weight) {
    const int M = input.size(0);
    const int K = input.size(1);
    const int N = weight.size(0);

    auto output = torch::empty({M, N}, input.options());

    // Use 16x16 threads per block
    const int threads = 16;
    dim3 threads(threads, threads);
    dim3 blocks((M + threads - 1) / threads,
                (N + threads - 1) / threads);

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "matmul_shared_kernel", ([&] {
        matmul_shared_kernel<scalar_t><<<blocks, threads>>>(
            output.data_ptr<scalar_t>(),
            input.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            M, N, K
        );
    }));

    return output;
}"""

cpp_source = """torch::Tensor matmul_cuda(torch::Tensor input, torch::Tensor weight);"""

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
