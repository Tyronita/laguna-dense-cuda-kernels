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
    const int N) {
    
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    const int col = blockIdx.y * blockDim.y + threadIdx.y;
    scalar_t sum = 0;
    
    // Shared memory tiles
    __shared__ scalar_t tileA[16][16];
    __shared__ scalar_t tileB[16][16];
    
    // Loop over tiles
    for (int t = 0; t < (N + 15) / 16; ++t) {
        // Collaborative loading of tiles into shared memory
        if (t * 16 + threadIdx.y < N && t * 16 + threadIdx.x < N) {
            tileA[threadIdx.y][threadIdx.x] = input[row * N + t * 16 + threadIdx.x];
        } else {
            tileA[threadIdx.y][threadIdx.x] = 0;
        }
        
        if (t * 16 + threadIdx.x < N && col < N) {
            tileB[threadIdx.y][threadIdx.x] = weight[col * N + t * 16 + threadIdx.x];
        } else {
            tileB[threadIdx.y][threadIdx.x] = 0;
        }
        
        __syncthreads();
        
        // Compute partial results for this tile
        #pragma unroll
        for (int k = 0; k < 16; ++k) {
            sum = fma(tileA[k][threadIdx.x], tileB[threadIdx.y][k], sum);
        }
        
        __syncthreads();
    }
    
    if (row < N && col < N) {
        output[row * N + col] = sum;
    }
}

// PyTorch wrapper function
torch::Tensor matmul_cuda(torch::Tensor input, torch::Tensor weight) {
    const int N = input.size(0);
    auto output = torch::empty_like(input);
    
    const int BLOCK_SIZE = 16;
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks((N + BLOCK_SIZE - 1) / BLOCK_SIZE,
                (N + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "matmul_shared_kernel", ([&] {
        matmul_shared_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            N
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
