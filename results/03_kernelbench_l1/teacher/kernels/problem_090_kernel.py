import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cumprod_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void cumprod_kernel(const float* input, float* output, int* shape, int num_dims, int dim, int total_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;
    
    // Compute multi-dimensional indices
    int indices[MAX_DIMS];
    int temp = idx;
    for (int i = num_dims - 1; i >= 0; i--) {
        indices[i] = temp % shape[i];
        temp /= shape[i];
    }
    
    // Compute linear index for the starting position along dim
    int linear_idx = 0;
    for (int i = 0; i < num_dims; i++) {
        if (i == dim) continue;
        linear_idx = linear_idx * shape[i] + indices[i];
    }
    
    // Compute stride for dim
    int stride = 1;
    for (int i = dim + 1; i < num_dims; i++) {
        stride *= shape[i];
    }
    
    // Compute position along dim
    int pos = indices[dim];
    
    // Compute base index
    int base_idx = linear_idx;
    for (int i = dim + 1; i < num_dims; i++) {
        base_idx = base_idx * shape[i] + indices[i];
    }
    
    // Recompute properly
    int base = idx;
    int stride_dim = 1;
    for (int i = dim + 1; i < num_dims; i++) {
        stride_dim *= shape[i];
    }
    
    int pos_in_dim = (idx / stride_dim) % shape[dim];
    int base_idx_new = idx - pos_in_dim * stride_dim;
    
    float result = 1.0f;
    for (int i = 0; i <= pos; i++) {
        int idx_i = base_idx_new + i * stride_dim;
        result *= input[idx_i];
    }
    output[idx] = result;
}

torch::Tensor cumprod_cuda(torch::Tensor input, int dim) {
    auto shape = input.sizes();
    int num_dims = input.dim();
    int total_elements = input.numel();
    
    // Handle negative dim
    if (dim < 0) {
        dim += num_dims;
    }
    
    auto output = torch::zeros_like(input);
    
    const int block_size = 256;
    const int num_blocks = (total_elements + block_size - 1) / block_size;
    
    // Allocate temporary storage for shape
    int* d_shape;
    cudaMalloc(&d_shape, num_dims * sizeof(int));
    cudaMemcpy(d_shape, shape.data(), num_dims * sizeof(int), cudaMemcpyHostToDevice);
    
    cumprod_kernel<<<num_blocks, block_size>>>(input.data_ptr<float>(), output.data_ptr<float>(), d_shape, num_dims, dim, total_elements);
    
    cudaFree(d_shape);
    return output;
}
"""

cumprod_cpp_source = """
torch::Tensor cumprod_cuda(torch::Tensor input, int dim);
"""

cumprod = load_inline(
    name="cumprod",
    cpp_sources=cumprod_cpp_source,
    cuda_sources=cumprod_source,
    functions=["cumprod_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    """
    A model that performs a cumulative product operation along a specified dimension.

    Parameters:
        dim (int): The dimension along which to perform the cumulative product operation.
    """

    def __init__(self, dim):
        """
        Initialize the CumulativeProductModel.

        Args:
            dim (int): The dimension along which to perform the cumulative product.
        """
        super(ModelNew, self).__init__()
        self.dim = dim
        self.cumprod = cumprod

    def forward(self, x):
        """
        Forward pass, computing the cumulative product along the specified dimension.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, *input_shape).

        Returns:
            torch.Tensor: Tensor of the same shape as `x` after applying cumulative product along `dim`.
        """
        return self.cumprod.cumprod_cuda(x, self.dim)