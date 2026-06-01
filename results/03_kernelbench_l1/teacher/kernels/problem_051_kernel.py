import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

argmax_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void argmax_kernel(const float* input, int* output, int batch_size, int dim1, int dim2, int dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size) {
        int max_idx = 0;
        float max_val = input[idx * dim1 * dim2 + 0 * dim2 + 0];
        for (int i = 1; i < dim2; i++) {
            float val = input[idx * dim1 * dim2 + 0 * dim2 + i];
            if (val > max_val) {
                max_val = val;
                max_idx = i;
            }
        }
        output[idx] = max_idx;
    }
}

torch::Tensor argmax_cuda(torch::Tensor input, int dim) {
    auto batch_size = input.size(0);
    auto dim1 = input.size(1);
    auto dim2 = input.size(2);
    auto output = torch::zeros({batch_size}, input.options().dtype(torch.int64));
    const int block_size = 256;
    const int num_blocks = (batch_size + block_size - 1) / block_size;
    argmax_kernel<<<num_blocks, block_size>>>(input.data_ptr<float>(), output.data_ptr<int64_t>(), batch_size, dim1, dim2, dim);
    return output;
}
"""
argmax_cpp_source = "torch::Tensor argmax_cuda(torch::Tensor input, int dim);"
argmax = load_inline(name="argmax", cpp_sources=argmax_cpp_source, cuda_sources=argmax_source, functions=["argmax_cuda"], verbose=True)

class ModelNew(nn.Module):
    def __init__(self, dim: int):
        super(ModelNew, self).__init__()
        self.dim = dim
        self.argmax = argmax

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.argmax.argmax_cuda(x, self.dim)