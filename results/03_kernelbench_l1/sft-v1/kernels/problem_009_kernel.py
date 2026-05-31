import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = """#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// CUDA kernel for matrix multiplication with transposed inputs
template <typename scalar_t>
__global__ void matmul_transpose_kernel(
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    const int M,
    const int N,
    const int K) {
    
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    const int col = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (row < M && col < N) {
        scalar_t sum = 0;
        for (int k = 0; k < K; ++k) {
            sum += input[col * M + row] * weight[k * M + row];
        }
        output[row * N + col] = sum;
    }
}

// PyTorch wrapper function
torch::Tensor matmul_transpose_cuda(torch::Tensor input, torch::Tensor weight) {
    const int M = input.size(0);
    const int N = weight.size(0);
    const int K = input.size(1);

    auto output = torch::empty({M, N}, input.options());

    const int threads = 16;
    dim3 threads(threads, threads);
    dim3 blocks((M + threads - 1) / threads,
                (N + threads - 1) / threads);

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "matmul_transpose_kernel", ([&] {
        matmul_transpose_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            M, N, K
        );
    }));

    return output;
}"""

cpp_source = """torch::Tensor matmul_transpose_cuda(torch::Tensor input, torch::Tensor weight);"""

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
