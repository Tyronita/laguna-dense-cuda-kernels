import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = """#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void argmax_kernel(const float* input, float* output, int size, int dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        int current_dim = dim - 1;
        float max_val = -FLT_MAX;
        for (int i = 0; i < current_dim; i++) {
            float val = input[idx * current_dim + i];
            if (val > max_val) {
                max_val = val;
            }
        }
        output[idx] = max_val;
    }
}

torch::Tensor argmax_cuda(torch::Tensor x, int dim) {
    auto size = x.numel();
    auto output = torch::empty_like(x);
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;
    argmax_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), output.data_ptr<float>(), size, dim);
    return output;
}"""

cpp_source = """torch::Tensor argmax_cuda(torch::Tensor x, int dim);"""

custom_ops = load_inline(
    name="custom_ops",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['argmax_cuda'],
    verbose=False,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"],
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x):
        return custom_ops.argmax_cuda(x)
