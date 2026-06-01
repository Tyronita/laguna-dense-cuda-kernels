import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

argmin_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void argmin_kernel(const float* input, int* output, int dim_size, int other_size, int dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < other_size) {
        int min_idx = 0;
        float min_val = input[idx * dim_size];
        for (int i = 1; i < dim_size; i++) {
            float val = input[idx * dim_size + i];
            if (val < min_val) {
                min_val = val;
                min_idx = i;
            }
        }
        output[idx] = min_idx;
    }
}

torch::Tensor argmin_cuda(torch::Tensor input, int dim) {
    auto input_shape = input.sizes();
    int dim_size = input_shape[dim];
    int other_size = 1;
    for (int i = 0; i < input_shape.size(); i++) {
        if (i != dim) {
            other_size *= input_shape[i];
        }
    }
    auto output = torch::zeros({other_size}, input.options().dtype(torch::kInt64));
    const int block_size = 256;
    const int num_blocks = (other_size + block_size - 1) / block_size;
    argmin_kernel<<<num_blocks, block_size>>>(input.data_ptr<float>(), output.data_ptr<int64_t>(), dim_size, other_size, dim);
    return output;
}
"""
argmin_cpp_source = "torch::Tensor argmin_cuda(torch::Tensor input, int dim);"
argmin = load_inline(name="argmin", cpp_sources=argmin_cpp_source, cuda_sources=argmin_source, functions=["argmin_cuda"], verbose=True)

class ModelNew(nn.Module):
    """
    Simple model that finds the index of the minimum value along a specified dimension.
    """
    def __init__(self, dim: int):
        """
        Initializes the model with the dimension to perform argmin on.

        Args:
            dim (int): Dimension along which to find the minimum value.
        """
        super(ModelNew, self).__init__()
        self.dim = dim
        self.argmin = argmin

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Finds the index of the minimum value along the specified dimension.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Tensor containing the indices of the minimum values along the specified dimension.
        """
        return self.argmin.argmin_cuda(x, self.dim)