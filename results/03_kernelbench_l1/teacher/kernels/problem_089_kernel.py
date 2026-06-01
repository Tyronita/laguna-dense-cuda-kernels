import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cumsum_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void cumsum_kernel(const float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = input[idx];
    }
    __syncthreads();
    
    for (int stride = 1; stride < gridDim.x; stride *= 2) {
        if (idx >= stride && idx < n) {
            output[idx] += output[idx - stride];
        }
        __syncthreads();
    }
}

torch::Tensor cumsum_cuda(torch::Tensor x, int dim) {
    auto input = x.contiguous();
    auto output = torch::zeros_like(input);
    int64_t n = input.numel();
    
    const int block_size = 256;
    const int num_blocks = (n + block_size - 1) / block_size;
    
    cumsum_kernel<<<num_blocks, block_size>>>(input.data_ptr<float>(), output.data_ptr<float>(), n);
    
    return output;
}
"""

cumsum_cpp_source = "torch::Tensor cumsum_cuda(torch::Tensor x, int dim);"

cumsum = load_inline(
    name="cumsum",
    cpp_sources=cumsum_cpp_source,
    cuda_sources=cumsum_source,
    functions=["cumsum_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    """
    A simple model that performs a cumulative sum (prefix sum) operation along a specified dimension.

    Parameters:
        dim (int): The dimension along which to perform the scan operation.
    """

    def __init__(self, dim):
        """
        Initialize the Scan model.

        Args:
            dim (int): The dimension along which to perform the cumulative sum.
        """
        super(ModelNew, self).__init__()
        self.dim = dim
        self.cumsum = cumsum

    def forward(self, x):
        """
        Forward pass for the Scan model, computing the cumulative sum along the specified dimension.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, *input_shape), where `*input_shape` 
                              can vary depending on the use case.

        Returns:
            torch.Tensor: Tensor of the same shape as `x` after applying cumulative sum along `dim`.
        """
        return self.cumsum.cumsum_cuda(x, self.dim)