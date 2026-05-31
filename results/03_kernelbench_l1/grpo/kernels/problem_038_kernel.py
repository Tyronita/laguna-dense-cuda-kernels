import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = """#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void l1_norm_kernel(const float* input, float* output, int size, int dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        int d = idx / dim;
        float sum = 0.0f;
        for (int i = 0; i < dim; i++) {
            sum += input[idx * dim + i] * input[idx * dim + i];
        }
        output[idx] = input[idx] / (sum / dim);
    }
}

torch::Tensor l1_norm_cuda(torch::Tensor x) {
    auto size = x.numel();
    auto dim = x.size(1);
    auto output = torch::empty_like(x);
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;
    l1_norm_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), output.data_ptr<float>(), size, dim);
    return output;
}"""

cpp_source = """torch::Tensor l1_norm_cuda(torch::Tensor x);"""

custom_ops = load_inline(
    name="custom_ops",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['l1_norm_cuda'],
    verbose=False,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"],
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x):
        return custom_ops.l1_norm_cuda(x)
