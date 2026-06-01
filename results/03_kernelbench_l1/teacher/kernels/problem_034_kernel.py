import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

instance_norm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void elementwise_sub_kernel(const float* input, float* output, int size, float mean) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) { output[idx] = input[idx] - mean; }
}

__global__ void elementwise_div_kernel(float* input, float* output, int size, float inv_std) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) { output[idx] = input[idx] * inv_std; }
}

__global__ void elementwise_add_bias_kernel(float* input, float* output, const float* weight, const float* bias, int C, int H, int W) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = C * H * W;
    if (idx < total) {
        int c = idx / (H * W);
        output[idx] = input[idx] * weight[c] + bias[c];
    }
}

torch::Tensor instance_norm_cuda(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, double eps) {
    auto B = x.size(0);
    auto C = x.size(1);
    auto H = x.size(2);
    auto W = x.size(3);
    auto HxW = H * W;
    
    auto out = torch::zeros_like(x);
    const int block_size = 256;
    
    for (int b = 0; b < B; b++) {
        // Compute mean and variance
        double sum = 0.0;
        int total = C * H * W;
        for (int i = 0; i < total; i++) {
            sum += x[b][0][i / H][i % H];
        }
        double mean = sum / total;
        
        double var_sum = 0.0;
        for (int i = 0; i < total; i++) {
            double diff = x[b][0][i / H][i % H] - mean;
            var_sum += diff * diff;
        }
        double variance = var_sum / total;
        double inv_std = 1.0 / sqrt(variance + eps);
        
        // Subtract mean
        for (int i = 0; i < total; i++) {
            int c = i / HxW;
            int h = (i / W) % H;
            int w = i % W;
            out[b][c][h][w] = x[b][c][h][w] - mean;
        }
        
        // Divide by std
        for (int i = 0; i < total; i++) {
            int c = i / HxW;
            int h = (i / W) % H;
            int w = i % W;
            out[b][c][h][w] = out[b][c][h][w] * inv_std;
        }
        
        // Apply weight and bias
        for (int i = 0; i < total; i++) {
            int c = i / HxW;
            int h = (i / W) % H;
            int w = i % W;
            out[b][c][h][w] = out[b][c][h][w] * weight[c] + bias[c];
        }
    }
    
    return out;
}
"""

instance_norm_cpp_source = "torch::Tensor instance_norm_cuda(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, double eps);"

instance_norm = load_inline(
    name="instance_norm",
    cpp_sources=instance_norm_cpp_source,
    cuda_sources=instance_norm_source,
    functions=["instance_norm_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    """
    Simple model that performs Instance Normalization.
    """
    def __init__(self, num_features: int):
        """
        Initializes the InstanceNorm layer.

        Args:
            num_features (int): Number of features in the input tensor.
        """
        super(ModelNew, self).__init__()
        self.num_features = num_features
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.eps = 1e-5
        self.instance_norm_cuda = instance_norm.instance_norm_cuda

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Instance Normalization to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features, height, width).

        Returns:
            torch.Tensor: Output tensor with Instance Normalization applied, same shape as input.
        """
        return self.instance_norm_cuda(x, self.weight, self.bias, self.eps)