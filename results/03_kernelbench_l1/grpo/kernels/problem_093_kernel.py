import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = """#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void masked_cumsum_kernel(const float* x, const float* mask, float* out, int size, int dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        int d = dim % 2;
        int base = idx * 2;
        float sum = 0.0f;
        for (int i = 0; i < 2; i++) {
            int pos = base + i;
            if (pos < size) {
                sum += x[pos] * mask[pos];
            }
        }
        out[idx] = sum;
    }
}

torch::Tensor masked_cumsum_cuda(torch::Tensor x, torch::Tensor mask, int dim) {
    auto size = x.numel();
    auto out = torch::zeros_like(x);
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;
    masked_cumsum_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), mask.data_ptr<float>(), out.data_ptr<float>(), size, dim);
    return out;
}"""

cpp_source = """torch::Tensor masked_cumsum_cuda(torch::Tensor x, torch::Tensor mask, int dim);"""

custom_ops = load_inline(
    name="custom_ops",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['masked_cumsum_cuda'],
    verbose=False,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"],
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x, mask):
        return custom_ops.masked_cumsum_cuda(x, mask)
