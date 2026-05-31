import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

group_norm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void group_norm_kernel(const float* x, float* out, float* mean, float* inv_var, 
                                   int batch_size, int num_features, int spatial_size, int num_groups,
                                   float eps, float* weight, float* bias) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_size = batch_size * num_features * spatial_size;
    if (idx < total_size) {
        int s_idx = idx % spatial_size;
        int f_idx = (idx / spatial_size) % num_features;
        int b_idx = idx / (spatial_size * num_features);
        
        float m = mean[b_idx * num_features + f_idx];
        float iv = inv_var[b_idx * num_features + f_idx];
        float w = weight[f_idx];
        float b = bias[f_idx];
        
        float centered = x[idx] - m;
        out[idx] = (centered * iv) * w + b;
    }
}

__global__ void compute_mean_inv_var_kernel(const float* x, float* mean, float* inv_var,
                                             int batch_size, int num_features, int spatial_size, int num_groups, float eps) {
    int g_idx = blockIdx.x;
    int b_idx = blockIdx.y;
    if (g_idx < num_groups && b_idx < batch_size) {
        int features_per_group = num_features / num_groups;
        int start_f = g_idx * features_per_group;
        int end_f = (g_idx + 1) * features_per_group;
        
        float sum = 0.0f;
        float sum_sq = 0.0f;
        int count = 0;
        
        for (int f = start_f; f < end_f; f++) {
            for (int s = 0; s < spatial_size; s++) {
                float val = x[b_idx * num_features * spatial_size + f * spatial_size + s];
                sum += val;
                sum_sq += val * val;
                count++;
            }
        }
        
        float m = sum / count;
        float variance = (sum_sq / count) - (m * m);
        float iv = 1.0f / sqrtf(variance + eps);
        
        mean[b_idx * num_features + (start_f + threadIdx.x)] = m;
        inv_var[b_idx * num_features + (start_f + threadIdx.x)] = iv;
    }
}

torch::Tensor group_norm_cuda(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, 
                             int num_groups, float eps) {
    auto batch_size = x.size(0);
    auto num_features = x.size(1);
    auto spatial_size = x.numel() / (batch_size * num_features);
    
    auto mean = torch::zeros({batch_size, num_features}, x.options());
    auto inv_var = torch::zeros({batch_size, num_features}, x.options());
    
    dim3 block(256);
    dim3 grid_mean_inv_var(num_groups, batch_size);
    compute_mean_inv_var_kernel<<<grid_mean_inv_var, block>>>(
        x.data_ptr<float>(), mean.data_ptr<float>(), inv_var.data_ptr<float>(),
        batch_size, num_features, spatial_size, num_groups, eps);
    
    auto out = torch::empty_like(x);
    dim3 grid_out((x.numel() + 255) / 256);
    group_norm_kernel<<<grid_out, block>>>(
        x.data_ptr<float>(), out.data_ptr<float>(), 
        mean.data_ptr<float>(), inv_var.data_ptr<float>(),
        batch_size, num_features, spatial_size, num_groups, eps,
        weight.data_ptr<float>(), bias.data_ptr<float>());
    
    return out;
}
"""

group_norm_cpp_source = """
torch::Tensor group_norm_cuda(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, 
                             int num_groups, float eps);
"""

group_norm = load_inline(
    name="group_norm",
    cpp_sources=group_norm_cpp_source,
    cuda_sources=group_norm_source,
    functions=["group_norm_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    """
    Simple model that performs Group Normalization with custom CUDA operators.
    """
    def __init__(self, num_features: int, num_groups: int):
        """
        Initializes the GroupNorm layer.

        Args:
            num_features (int): Number of features in the input tensor.
            num_groups (int): Number of groups to divide the channels into.
        """
        super(ModelNew, self).__init__()
        self.num_features = num_features
        self.num_groups = num_groups
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.eps = 1e-5
        self.group_norm_cuda = group_norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Group Normalization to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features, *).

        Returns:
            torch.Tensor: Output tensor with Group Normalization applied, same shape as input.
        """
        return self.group_norm_cuda.group_norm_cuda(x, self.weight, self.bias, self.num_groups, self.eps)