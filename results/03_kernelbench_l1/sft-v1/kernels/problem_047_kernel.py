import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = """#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// CUDA kernel for sum reduction over a given dimension
// Each thread computes one element of the output tensor
// The input tensor is assumed to be 3D with shape (..., dim, ...)
// and the output tensor is assumed to be 3D with shape (..., 1, ...)

template <typename scalar_t>
__global__ void sum_reduce_kernel(
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ input,
    const int64_t batch_size,
    const int64_t dim1,
    const int64_t dim2,
    const int64_t reduce_dim) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size * dim2) {
        int batch_idx = idx / dim2;
        int dim_idx = idx % dim2;
        scalar_t sum = 0;
        for (int d = 0; d < dim1; ++d) {
            sum += input[batch_idx * dim1 + d * dim2 + dim_idx];
        }
        output[batch_idx * dim2 + dim_idx] = sum;
    }
}

// PyTorch wrapper function
torch::Tensor forward_cuda(torch::Tensor input, int64_t reduce_dim) {
    // Ensure input is a 3D tensor
    TORCH_CHECK(input.dim() == 3, "Input must be 3D");
    TORCH_CHECK(input.size(2) == reduce_dim, "Input dimension must be reduce_dim");

    const int64_t batch_size = input.size(0);
    const int64_t dim1 = input.size(1);
    const int64_t dim2 = input.size(2);

    auto output = torch::empty({batch_size, 1, dim2}, input.options());

    const int threads = 256;
    const int blocks = (batch_size * dim2 + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "sum_reduce_kernel", ([&] {
        sum_reduce_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            batch_size,
            dim1,
            dim2,
            reduce_dim
        );
    }));

    return output;
}"""

cpp_source = """torch::Tensor forward_cuda(torch::Tensor input, int64_t reduce_dim);"""

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

    def forward(self, x):
        return custom_ops.forward_cuda(x)
