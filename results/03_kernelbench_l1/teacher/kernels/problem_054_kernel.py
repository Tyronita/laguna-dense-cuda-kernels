import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

conv3d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

template<typename T>
__device__ T clamp(T val, T min_val, T max_val) {
    return val < min_val ? min_val : (val > max_val ? max_val : val);
}

__global__ void conv3d_kernel(const float* input, const float* weight, const float* bias, float* output,
                              int batch_size, int in_channels, int out_channels,
                              int depth, int width, int height,
                              int kernel_size, int stride, int padding, int dilation, int groups) {
    
    int out_depth = (depth + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    int out_width = (width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    int out_height = (height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    
    int total_elements = batch_size * out_channels * out_depth * out_width * out_height;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= total_elements) return;
    
    int temp = idx;
    int out_n = temp / (out_channels * out_depth * out_width * out_height);
    temp %= (out_channels * out_depth * out_width * out_height);
    int out_c = temp / (out_depth * out_width * out_height);
    temp %= (out_depth * out_width * out_height);
    int out_d = temp / (out_width * out_height);
    temp %= (out_width * out_height);
    int out_h = temp / out_height;
    int out_w = temp % out_height;
    
    int in_c_group = out_c / (out_channels / groups);
    int in_c_start = in_c_group * (in_channels / groups);
    int in_c_end = in_c_start + (in_channels / groups);
    
    float sum = 0.0f;
    
    for (int kc = 0; kc < kernel_size; kc++) {
        for (int kr = 0; kr < kernel_size; kr++) {
            for (int kw = 0; kw < kernel_size; kw++) {
                int in_d = out_d * stride - padding + kc * dilation;
                int in_r = out_h * stride - padding + kr * dilation;
                int in_w = out_w * stride - padding + kw * dilation;
                
                if (in_d >= 0 && in_d < depth && in_r >= 0 && in_r < width && in_w >= 0 && in_w < height) {
                    for (int ic = in_c_start; ic < in_c_end; ic++) {
                        int weight_idx = ((out_c - in_c_start * out_channels / groups) * in_channels + ic) * kernel_size * kernel_size * kernel_size + 
                                         (kc * kernel_size + kr) * kernel_size + kw;
                        int in_idx = ((out_n * in_channels + ic) * depth + in_d) * width + in_r;
                        in_idx = in_idx * height + in_w;
                        sum += input[in_idx] * weight[weight_idx];
                    }
                }
            }
        }
    }
    
    if (bias != nullptr) {
        sum += bias[out_c];
    }
    
    int out_idx = ((out_n * out_channels + out_c) * out_depth + out_d) * out_width + out_h;
    out_idx = out_idx * out_height + out_w;
    output[out_idx] = sum;
}

torch::Tensor conv3d_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
                          int stride, int padding, int dilation, int groups) {
    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto depth = input.size(2);
    auto width = input.size(3);
    auto height = input.size(4);
    auto out_channels = weight.size(0);
    
    int out_depth = (depth + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    int out_width = (width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    int out_height = (height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    
    auto output = torch::zeros({batch_size, out_channels, out_depth, out_width, out_height}, input.options());
    
    const int block_size = 256;
    int total_elements = batch_size * out_channels * out_depth * out_width * out_height;
    const int num_blocks = (total_elements + block_size - 1) / block_size;
    
    const float* bias_ptr = bias.defined() ? bias.data_ptr<float>() : nullptr;
    
    conv3d_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), bias_ptr, output.data_ptr<float>(),
        batch_size, in_channels, out_channels, depth, width, height,
        weight.size(2), stride, padding, dilation, groups
    );
    
    return output;
}
"""

conv3d_cpp_source = """
torch::Tensor conv3d_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
                          int stride, int padding, int dilation, int groups);
"""

conv3d = load_inline(
    name="conv3d",
    cpp_sources=conv3d_cpp_source,
    cuda_sources=conv3d_source,
    functions=["conv3d_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        
        # Create weight and bias parameters
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels // groups, kernel_size, kernel_size, kernel_size))
        if bias:
            self.bias_param = nn.Parameter(torch.randn(out_channels))
        else:
            self.bias_param = None
        
        self.conv3d = conv3d
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias = self.bias_param if self.bias else torch.Tensor()
        return self.conv3d.conv3d_cuda(x, self.weight, bias, self.stride, self.padding, self.dilation, self.groups)