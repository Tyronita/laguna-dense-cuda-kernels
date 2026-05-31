import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = """#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void optimized_cumsum_kernel(const float* input, float* output, int size, int dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        int d = dim % 2;  // Assuming dim is divisible by 2
        int offset = (idx / 2) * 2;
        float sum = 0.0f;
        #pragma unroll
        for (int i = 0; i < 2; i++) {
            sum += input[offset + i];
        }
        output[idx] = sum;
    }
}

torch::Tensor optimized_cumsum_cuda(torch::Tensor x, int dim) {
    TORCH_CHECK(x.dim() == 2, "Input must be 2D");
    TORCH_CHECK(x.size(1) == dim, "Dimension mismatch");
    TORCH_CHECK(x.is_cuda(), "Input must be on CUDA");
    TORCH_CHECK(x.is_contiguous(), "Input must be contiguous");

    int size = x.size(0);
    auto output = torch::empty_like(x);
    
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;
    
    optimized_cumsum_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), output.data_ptr<float>(), size, dim);
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "Kernel failed: ", cudaGetErrorString(err));
    
    return output;
}"""

cpp_source = """torch::Tensor optimized_cumsum_cuda(torch::Tensor x, int dim);"""

custom_ops = load_inline(
    name="custom_ops",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['optimized_cumsum_cuda'],
    verbose=False,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"],
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x):
        return custom_ops.optimized_cumsum_cuda(x)
