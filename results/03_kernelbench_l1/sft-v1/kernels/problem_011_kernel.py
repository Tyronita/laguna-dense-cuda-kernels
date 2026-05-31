import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = """#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// CUDA kernel for 4D tensor-matrix multiplication with loop unrolling
template <typename scalar_t>
__global__ void einsum_unroll_kernel(
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    const int64_t BATCH,
    const int64_t I,
    const int64_t J,
    const int64_t L,
    const int64_t K) {
    
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_elements = BATCH * I * J * K;
    
    if (idx < total_elements) {
        const int j = idx % J;
        const int k = (idx / (J * I)) % K;
        const int i = (idx / (J * I)) % I;
        const int b = idx / (J * I);
        
        scalar_t sum = 0;
        #pragma unroll
        for (int l = 0; l < L; ++l) {
            const int input_idx = ((b * I * J + i * J + j) * L + l);
            const int weight_idx = l * K + k;
            sum += input[input_idx] * weight[weight_idx];
        }
        output[idx] = sum;
    }
}

// PyTorch wrapper function
torch::Tensor forward_cuda(torch::Tensor input, torch::Tensor weight) {
    const int64_t BATCH = input.size(0);
    const int64_t I = input.size(1);
    const int64_t J = input.size(2);
    const int64_t L = input.size(3);
    const int64_t K = weight.size(1);

    auto output = torch::empty({BATCH, I, J, K}, input.options());

    const int threads = 256;
    const int blocks = (BATCH * I * J * K + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "einsum_unroll_kernel", ([&] {
        einsum_unroll_kernel<scalar_t><<<blocks, threads>>>(
            output.data_ptr<scalar_t>(),
            input.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            BATCH, I, J, L, K
        );
    }));

    return output;
}"""

cpp_source = """torch::Tensor forward_cuda(torch::Tensor input, torch::Tensor weight);"""

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
