import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = """#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// CUDA kernel for matrix multiplication with optimized memory access
template <typename scalar_t>
__global__ void triangular_mm_kernel(
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ A,
    const scalar_t* __restrict__ B,
    const int N) {
    
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    const int col = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (row < N || col < N) {
        if (row >= col) {
            output[row * N + col] = 0.0f;
        } else {
            scalar_t sum = 0.0f;
            for (int k = col; k <= row; ++k) {
                sum += A[row * N + k] * B[k * N + col];
            }
            output[row * N + col] = sum;
        }
    }
}

// PyTorch wrapper function
torch::Tensor triangular_mm_cuda(torch::Tensor A, torch::Tensor B) {
    const int N = A.size(0);
    auto output = torch::empty_like(A);
    
    const int threads = 16;
    dim3 threads(threads, threads);
    dim3 blocks(
        (N + threads - 1) / threads,
        (N + threads - 1) / threads
    );

    AT_DISPATCH_FLOATING_TYPES(A.type(), "triangular_mm_kernel", ([&] {
        triangular_mm_kernel<scalar_t><<<blocks, threads>>>(
            output.data_ptr<scalar_t>(),
            A.data_ptr<scalar_t>(),
            B.data_ptr<scalar_t>(),
            N
        );
    }));

    return output;
}"""

cpp_source = """torch::Tensor triangular_mm_cuda(torch::Tensor A, torch::Tensor B);"""

custom_ops = load_inline(
    name="custom_ops",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['triangular_mm_cuda'],
    verbose=False,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"],
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, A, B):
        return custom_ops.triangular_mm_cuda(A, B)
