import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

mean_reduction_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void mean_kernel(const float* input, float* output, int size, int dim_size, int dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        int total_elements = 1;
        int stride = 1;
        for (int i = 0; i < (int)input.dim(); i++) {
            if (i != dim) {
                stride *= input.size(i);
            }
        }
        
        int outer_idx = idx / dim_size;
        int inner_idx = idx % dim_size;
        
        float sum = 0.0f;
        for (int i = 0; i < dim_size; i++) {
            int linear_idx = outer_idx * dim_size * dim_size + i * dim_size + inner_idx;
            sum += input[linear_idx];
        }
        output[idx] = sum / dim_size;
    }
}

torch::Tensor mean_cuda(torch::Tensor input, int dim) {
    auto input_shape = input.sizes();
    int dim_size = input.size(dim);
    std::vector<int64_t> output_shape;
    for (int i = 0; i < input.dim(); i++) {
        if (i != dim) {
            output_shape.push_back(input.size(i));
        }
    }
    
    auto output = torch::zeros(output_shape, input.options());
    int size = 1;
    for (auto s : output_shape) {
        size *= s;
    }
    
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;
    mean_kernel<<<num_blocks, block_size>>>(input.data_ptr<float>(), output.data_ptr<float>(), size, dim_size, dim);
    return output;
}
"""

mean_reduction_cpp_source = "torch::Tensor mean_cuda(torch::Tensor input, int dim);"
mean_reduction = load_inline(name="mean_reduction", cpp_sources=mean_reduction_cpp_source, cuda_sources=mean_reduction_source, functions=["mean_cuda"], verbose=True)

class ModelNew(nn.Module):
    """
    Simple model that performs mean reduction over a specific dimension.
    """
    def __init__(self, dim: int):
        """
        Initializes the model with the dimension to reduce over.

        Args:
            dim (int): The dimension to reduce over.
        """
        super(ModelNew, self).__init__()
        self.dim = dim
        self.mean_cuda = mean_reduction.mean_cuda

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reduces the input tensor along the specified dimension by taking the mean.

        Args:
            x (torch.Tensor): Input tensor of arbitrary shape.

        Returns:
            torch.Tensor: Output tensor with reduced dimension. The shape of the output is the same as the input except for the reduced dimension which is removed.
        """
        return self.mean_cuda(x, self.dim)