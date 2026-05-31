import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = """#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void optimized_cumsum_kernel(const float* input, float* output, int size, int dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float sum = 0.0f;
        for (int i = 0; i < dim; i++) {
            sum += input[idx * dim + i];
        }
        output[idx] = sum;
    }
}

torch::Tensor optimized_cumsum_cuda(torch::Tensor x, int dim) {
    auto size = x.numel();
    auto output = torch::empty_like(x);
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;
    optimized_cumsum_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), output.data_ptr<float>(), size, dim);
    return output;
}

torch::Tensor forward(torch::Tensor x, int dim) {
    return optimized_cumsum_cuda(x, dim).transpose(dim, 1).contiguous();
}"""

cpp_source = """torch::Tensor optimized_cumsum_cuda(torch::Tensor x, int dim);\ntorch::Tensor forward(torch::Tensor x, int dim);"""

custom_ops = load_inline(
    name="custom_ops",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['optimized_cumsum_cuda', 'forward'],
    verbose=False,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"],
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x):
        return custom_ops.optimized_cumsum_cuda(x)
