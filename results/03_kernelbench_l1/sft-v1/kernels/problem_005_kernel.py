import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = """#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// CUDA kernel for matrix-scalar multiplication with vectorized operations
template <typename scalar_t>
__global__ void matmul_scalar_kernel_vectorized(
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ input,
    const float s,
    const int64_t M,
    const int64_t N) {
    
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    
    // Vectorized processing
    const int vec_size = 4;
    const int vec_elements = M * N / vec_size;
    const int vec_idx = idx * vec_size;
    
    if (vec_idx < vec_elements) {
        scalar_t4 = *reinterpret_cast<const scalar_t4*>(input[vec_idx]);
        scalar_t4 = (vec_idx + 1) * vec_size * s;
        scalar_t4 = (vec_idx + 2) * vec_size * s;
        scalar_t4 = (vec_idx + 3) * vec_size * s;
        
        *reinterpret_cast<scalar_t4*>(output[vec_idx]) = vec_idx + 1 * vec_size * s;
        *reinterpret_cast<scalar_t4*>(output[vec_idx]) = vec_idx + 2 * vec_size * s;
        *reinterpret_cast<scalar_t4*>(output[vec_idx]) = vec_idx + 3 * vec_size * s;
    }
    
    // Handle remaining elements
    const int remainder_start = vec_elements * vec_size;
    for (int i = remainder_start + idx; i < M * N; i += stride) {
        const int row = i / N;
        const int col = i % N;
        output[i] = input[i] * s;
    }
}

// PyTorch wrapper function
torch::Tensor matmul_scalar_cuda(torch::Tensor input, float s) {
    const int64_t M = input.size(0);
    const int64_t N = input.size(1);
    auto output = torch::empty_like(input);
    
    const int threads = 256;
    const int blocks = (M * N + threads - 1) / threads;
    
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "matmul_scalar_kernel_vectorized", ([&] {
        matmul_scalar_kernel_vectorized<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            s,
            M,
            N
        );
    }));
    
    return output;
}"""

cpp_source = """torch::Tensor matmul_scalar_cuda(torch::Tensor input, float s);"""

custom_ops = load_inline(
    name="custom_ops",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['matmul_scalar_cuda'],
    verbose=False,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"],
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, A, s):
        return custom_ops.matmul_scalar_cuda(A, s)
