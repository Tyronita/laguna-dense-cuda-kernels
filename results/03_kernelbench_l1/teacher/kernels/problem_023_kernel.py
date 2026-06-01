import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

softmax_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void softmax_kernel(const float* input, float* output, int batch_size, int num_features) {
    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;
    
    if (batch_idx < batch_size) {
        int idx = batch_idx * num_features;
        float max_val = -INFINITY;
        
        // Find max for numerical stability
        for (int i = tid; i < num_features; i += blockDim.x) {
            max_val = fmaxf(max_val, input[idx + i]);
        }
        
        // Reduce max across threads
        for (int s = blockDim.x / 2; s >= 32; s /= 2) {
            max_val = fmaxf(max_val, __syncthreads_and(__syncthreads_or(0)) ? max_val : max_val);
        }
        
        // Shared memory for max
        extern __shared__ float s_max[];
        if (tid < 32) {
            s_max[tid] = max_val;
        }
        __syncthreads();
        
        if (tid < 32) {
            for (int s = blockDim.x / 64; s >= 1; s /= 2) {
                s_max[tid] = fmaxf(s_max[tid], s_max[tid + s]);
            }
        }
        __syncthreads();
        max_val = s_max[0];
        
        // Compute exp and sum
        float sum = 0.0f;
        for (int i = tid; i < num_features; i += blockDim.x) {
            float exp_val = expf(input[idx + i] - max_val);
            output[idx + i] = exp_val;
            sum += exp_val;
        }
        
        // Reduce sum
        for (int s = blockDim.x / 2; s >= 32; s /= 2) {
            sum += __syncthreads_or(0) ? sum : sum;
        }
        
        // Shared memory for sum
        extern __shared__ float s_sum[];
        if (tid < 32) {
            s_sum[tid] = sum;
        }
        __syncthreads();
        
        if (tid < 32) {
            for (int s = blockDim.x / 64; s >= 1; s /= 2) {
                s_sum[tid] = s_sum[tid] + s_sum[tid + s];
            }
        }
        __syncthreads();
        sum = s_sum[0];
        
        // Normalize
        for (int i = tid; i < num_features; i += blockDim.x) {
            output[idx + i] /= sum;
        }
    }
}

torch::Tensor softmax_cuda(torch::Tensor x) {
    auto batch_size = x.size(0);
    auto num_features = x.size(1);
    auto output = torch::zeros_like(x);
    
    const int block_size = 256;
    softmax_kernel<<<batch_size, block_size, 256 * sizeof(float)>>>(
        x.data_ptr<float>(), 
        output.data_ptr<float>(), 
        batch_size, 
        num_features
    );
    return output;
}
"""

softmax_cpp_source = "torch::Tensor softmax_cuda(torch::Tensor x);"

softmax = load_inline(
    name="softmax",
    cpp_sources=softmax_cpp_source,
    cuda_sources=softmax_source,
    functions=["softmax_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    """
    Simple model that performs a Softmax activation.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
        self.softmax = softmax

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Softmax activation to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features).

        Returns:
            torch.Tensor: Output tensor with Softmax applied, same shape as input.
        """
        return self.softmax.softmax_cuda(x)