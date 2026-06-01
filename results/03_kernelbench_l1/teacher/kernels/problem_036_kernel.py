import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

rms_norm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void rms_norm_kernel(const float* x, float* out, float eps, int num_features, int total_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_elements) {
        int batch_idx = idx / num_features;
        float sum_sq = 0.0f;
        for (int i = 0; i < num_features; i++) {
            int element_idx = batch_idx * num_features + i;
            sum_sq += x[element_idx] * x[element_idx];
        }
        float rms = sqrtf(sum_sq / num_features + eps);
        int element_idx = idx;
        out[element_idx] = x[element_idx] / rms;
    }
}

__global__ void rms_norm_reduction_kernel(const float* x, float* batch_sums, int num_features, int batch_size) {
    int batch_idx = blockIdx.x;
    if (batch_idx < batch_size) {
        float sum_sq = 0.0f;
        for (int i = 0; i < num_features; i++) {
            int element_idx = batch_idx * num_features + i;
            sum_sq += x[element_idx] * x[element_idx];
        }
        batch_sums[batch_idx] = sum_sq;
    }
}

torch::Tensor rms_norm_cuda(torch::Tensor x, float eps, int num_features) {
    auto batch_size = x.size(0);
    auto total_elements = x.numel();
    auto out = torch::zeros_like(x);
    
    auto batch_sums = torch::zeros({batch_size}, x.options());
    
    const int block_size = 256;
    const int num_blocks = (batch_size + block_size - 1) / block_size;
    rms_norm_reduction_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), batch_sums.data_ptr<float>(), num_features, batch_size);
    
    const int elementwise_block_size = 256;
    const int elementwise_num_blocks = (total_elements + elementwise_block_size - 1) / elementwise_block_size;
    rms_norm_kernel<<<elementwise_num_blocks, elementwise_block_size>>>(x.data_ptr<float>(), out.data_ptr<float>(), eps, num_features, total_elements);
    
    return out;
}
"""

rms_norm_cpp_source = "torch::Tensor rms_norm_cuda(torch::Tensor x, float eps, int num_features);"

rms_norm = load_inline(
    name="rms_norm",
    cpp_sources=rms_norm_cpp_source,
    cuda_sources=rms_norm_source,
    functions=["rms_norm_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    """
    Simple model that performs RMS Normalization.
    """
    def __init__(self, num_features: int, eps: float = 1e-5):
        """
        Initializes the RMSNorm layer.

        Args:
            num_features (int): Number of features in the input tensor.
            eps (float, optional): A small value added to the denominator to avoid division by zero. Defaults to 1e-5.
        """
        super(ModelNew, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.rms_norm = rms_norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies RMS Normalization to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features, *).

        Returns:
            torch.Tensor: Output tensor with RMS Normalization applied, same shape as input.
        """
        return self.rms_norm.rms_norm_cuda(x, self.eps, self.num_features)