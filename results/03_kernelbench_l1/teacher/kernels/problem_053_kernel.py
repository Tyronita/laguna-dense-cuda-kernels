import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

min_reduction_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void min_reduction_kernel(const float* input, float* output, int dim_size, int other_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < other_size) {
        float min_val = input[idx * dim_size];
        for (int i = 1; i < dim_size; i++) {
            float val = input[idx * dim_size + i];
            if (val < min_val) {
                min_val = val;
            }
        }
        output[idx] = min_val;
    }
}

torch::Tensor min_reduction_cuda(torch::Tensor x, int dim) {
    auto input_shape = x.sizes();
    int dim_size = input_shape[dim];
    int other_size = 1;
    for (int i = 0; i < input_shape.size(); i++) {
        if (i != dim) {
            other_size *= input_shape[i];
        }
    }
    
    auto output_shape = input_shape.vec();
    output_shape.erase(output_shape.begin() + dim);
    auto output = torch::zeros(output_shape, x.dtype(), x.device());
    
    const int block_size = 256;
    const int num_blocks = (other_size + block_size - 1) / block_size;
    min_reduction_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), output.data_ptr<float>(), dim_size, other_size);
    return output;
}
"""
min_reduction_cpp_source = "torch::Tensor min_reduction_cuda(torch::Tensor x, int dim);"
min_reduction = load_inline(name="min_reduction", cpp_sources=min_reduction_cpp_source, cuda_sources=min_reduction_source, functions=["min_reduction_cuda"], verbose=True)

class ModelNew(nn.Module):
    """
    Simple model that performs min reduction over a specific dimension.
    """
    def __init__(self, dim: int):
        """
        Initializes the model with the dimension to reduce over.

        Args:
            dim (int): The dimension to reduce over.
        """
        super(ModelNew, self).__init__()
        self.dim = dim
        self.min_reduction = min_reduction

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies min reduction over the specified dimension to the input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after min reduction over the specified dimension.
        """
        return self.min_reduction.min_reduction_cuda(x, self.dim)