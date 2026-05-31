import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = """#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// CUDA kernel for matrix-vector multiplication with loop unrolling
template <typename scalar_t>
__global__ void matvecmul_kernel_unrolled(
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ A,
    const scalar_t* __restrict__ B,
    const int M,
    const int K) {
    
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M) {
        scalar_t sum = 0;
        // Unroll the loop by a factor of 4
        int unroll_steps = (K / 4) * 4;
        for (int k = 0; k < unroll_steps; k += 4) {
            sum += A[row * K + k] * B[k];
            sum += A[row * K + k + 1] * B[k + 1];
            sum += A[row * K + k + 2] * B[k + 2];
            sum += A[row * K + k + 3] * B[k + 3];
        }
        // Handle remaining elements
        for (int k = unroll_steps; k < K; k++) {
            sum += A[row * K + k] * B[k];
        }
        output[row] = sum;
    }
}

// PyTorch wrapper function
torch::Tensor matvecmul_cuda(torch::Tensor A, torch::Tensor B) {
    const int M = A.size(0);
    const int K = A.size(1);
    auto output = torch::empty({M}, A.options());
    
    const int threads = 256;
    const int blocks = (M + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(A.type(), "matvecmul_kernel_unrolled", ([&] {
        matvecmul_kernel_unrolled<scalar_t><<<blocks, threads>>>(
            output.data_ptr<scalar_t>(),
            A.data_ptr<scalar_t>(),
            B.data_ptr<scalar_t>(),
            M,
            K
        );
    }));

    return output.view({M, 1});
}"""

cpp_source = """torch::Tensor matvecmul_cuda(torch::Tensor A, torch::Tensor B);"""

custom_ops = load_inline(
    name="custom_ops",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['matvecmul_cuda'],
    verbose=False,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"],
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, A, B):
        return custom_ops.matvecmul_cuda(A, B)
