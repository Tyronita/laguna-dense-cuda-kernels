import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = """#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// CUDA kernel for batched matrix multiplication with loop unrolling
template <typename scalar_t>
__global__ void bmm_unroll_kernel(
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ A,
    const scalar_t* __restrict__ B,
    const int batch_size,
    const int M,
    const int K,
    const int N) {
    
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    const int col = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_idx = blockIdx.z;
    
    if (row < M && col < N) {
        scalar_t sum = 0;
        for (int k = 0; k < K; ++k) {
            sum += A[batch_idx * M * K + row * K + k] * 
                   B[batch_idx * K * N + k * N + col];
        }
        output[batch_idx * M * N + row * N + col] = sum;
    }
}

// PyTorch wrapper function
torch::Tensor forward_cuda(torch::Tensor A, torch::Tensor B) {
    const int batch_size = A.size(0);
    const int M = A.size(1);
    const int K = A.size(2);
    const int N = B.size(1);

    auto C = torch::empty({batch_size, M, N}, A.options());

    // Use 32x32 threads per block for better occupancy
    const int threads = 32;
    dim3 threads(threads, threads);
    dim3 blocks(
        (M + threads - 1) / threads,
        (N + threads - 1) / threads,
        batch_size
    );

    AT_DISPATCH_FLOATING_TYPES(A.type(), "bmm_unroll_kernel", ([&] {
        bmm_unroll_kernel<scalar_t><<<blocks, threads>>>(
            C.data_ptr<scalar_t>(),
            A.data_ptr<scalar_t>(),
            B.data_ptr<scalar_t>(),
            batch_size, M, K, N
        );
    }));

    return C;
}"""

cpp_source = """torch::Tensor forward_cuda(torch::Tensor A, torch::Tensor B);"""

custom_ops = load_inline(
    name="custom_ops",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['forward_cuda'],
    verbose=False,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"],
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, A, B):
        return custom_ops.forward_cuda(A, B)
