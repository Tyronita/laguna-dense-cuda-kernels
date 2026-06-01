import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

conv3d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void conv3d_kernel(const float* input, const float* weight, const float* bias, float* output,
                              int batch_size, int in_channels, int out_channels,
                              int depth, int height, int width,
                              int kernel_d, int kernel_h, int kernel_w,
                              int stride, int padding, int dilation, int groups) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * out_channels * 
                         ((depth + 2 * padding - dilation * (kernel_d - 1) - 1) / stride + 1) *
                         ((height + 2 * padding - dilation * (kernel_h - 1) - 1) / stride + 1) *
                         ((width + 2 * padding - dilation * (kernel_w - 1) - 1) / stride + 1);
    
    if (idx >= total_elements) return;
    
    // Compute output indices
    int d_out = idx / (out_channels * ((height + 2 * padding - dilation * (kernel_h - 1) - 1) / stride + 1) * 
                       ((width + 2 * padding - dilation * (kernel_w - 1) - 1) / stride + 1));
    int remainder = idx % (out_channels * ((height + 2 * padding - dilation * (kernel_h - 1) - 1) / stride + 1) * 
                           ((width + 2 * padding - dilation * (kernel_w - 1) - 1) / stride + 1));
    int c_out = remainder / (((height + 2 * padding - dilation * (kernel_h - 1) - 1) / stride + 1) * 
                             ((width + 2 * padding - dilation * (kernel_w - 1) - 1) / stride + 1));
    remainder = remainder % (((height + 2 * padding - dilation * (kernel_h - 1) - 1) / stride + 1) * 
                             ((width + 2 * padding - dilation * (kernel_w - 1) - 1) / stride + 1));
    int h_out = remainder / (((width + 2 * padding - dilation * (kernel_w - 1) - 1) / stride + 1));
    int w_out = remainder % (((width + 2 * padding - dilation * (kernel_w - 1) - 1) / stride + 1));
    
    int n = idx / total_elements;
    
    float sum = 0;
    for (int kc = 0; kc < in_channels; kc++) {
        int c_in = kc / groups;
        int g = kc / in_channels;
        
        for (int kd = 0; kd < kernel_d; kd++) {
            int d_in = d_out * stride - padding + kd * dilation;
            if (d_in < 0 || d_in >= depth) continue;
            
            for (int kh = 0; kh < kernel_h; kh++) {
                int h_in = h_out * stride - padding + kh * dilation;
                if (h_in < 0 || h_in >= height) continue;
                
                for (int kw = 0; kw < kernel_w; kw++) {
                    int w_in = w_out * stride - padding + kw * dilation;
                    if (w_in < 0 || w_in >= width) continue;
                    
                    int weight_idx = ((kc * out_channels + c_out) * kernel_d + kd) * kernel_h + kh) * kernel_w + kw;
                    int input_idx = ((n * in_channels + c_in) * depth + d_in) * height * width + h_in * width + w_in;
                    sum += input[input_idx] * weight[weight_idx];
                }
            }
        }
    }
    
    int out_idx = ((n * out_channels + c_out) * 
                   ((depth + 2 * padding - dilation * (kernel_d - 1) - 1) / stride + 1) *
                   ((height + 2 * padding - dilation * (kernel_h - 1) - 1) / stride + 1) *
                 ((width + 2 * padding - dilation * (kernel_w - 1) - 1) / stride + 1)) +
                  d_out * ((height + 2 * padding - dilation * (kernel_h - 1) - 1) / stride + 1) *
                  ((width + 2 * padding - dilation * (kernel_w - 1) - 1) / stride + 1) +
                  h_out * ((width + 2 * padding - dilation * (kernel_w - 1) - 1) / stride + 1) +
                  w_out;
    
    if (bias != nullptr) {
        sum += bias[c_out];
    }
    output[out_idx] = sum;
}

torch::Tensor conv3d_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
                          int stride, int padding, int dilation, int groups) {
    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto depth = input.size(2);
    auto height = input.size(3);
    auto width = input.size(4);
    
    auto out_channels = weight.size(0);
    auto kernel_d = weight.size(2);
    auto kernel_h = weight.size(3);
    auto kernel_w = weight.size(4);
    
    auto out_depth = (depth + 2 * padding - dilation * (kernel_d - 1) - 1) / stride + 1;
    auto out_height = (height + 2 * padding - dilation * (kernel_h - 1) - 1) / stride + 1;
    auto out_width = (width + 2 * padding - dilation * (kernel_w - 1) - 1) / stride + 1;
    
    auto output = torch::zeros({batch_size, out_channels, out_depth, out_height, out_width}, input.options());
    
    auto total_elements = batch_size * out_channels * out_depth * out_height * out_width;
    
    const int block_size = 256;
    const int num_blocks = (total_elements + block_size - 1) / block_size;
    
    const float* bias_ptr = bias.defined() ? bias.data_ptr<float>() : nullptr;
    
    conv3d_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), bias_ptr, output.data_ptr<float>(),
        batch_size, in_channels, out_channels, depth, height, width,
        kernel_d, kernel_h, kernel_w, stride, padding, dilation, groups);
    
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
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
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
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels // groups, 
                                               kernel_size[0], kernel_size[1], kernel_size[2]))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_buffer('bias', torch.tensor([]))
        
        self.conv3d = conv3d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv3d.conv3d_cuda(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)