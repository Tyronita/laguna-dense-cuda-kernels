import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

conv_transpose2d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void conv_transpose2d_kernel(
    const float* input, 
    const float* weight, 
    const float* bias,
    float* output,
    int batch_size, int in_channels, int out_channels,
    int in_height, int in_width,
    int out_height, int out_width,
    int kernel_h, int kernel_w,
    int stride, int padding, int output_padding,
    int in_channel_groups, int out_channel_groups
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * out_channels * out_height * out_width;
    
    if (idx >= total_elements) return;
    
    int n = idx / (out_channels * out_height * out_width);
    int remainder = idx % (out_channels * out_height * out_width);
    int oc = remainder / (out_height * out_width);
    remainder = remainder % (out_height * out_width);
    int oh = remainder / out_width;
    int ow = remainder % out_width;
    
    int in_c_group = oc / out_channel_groups;
    int g = oc / out_channel_groups;
    
    float sum = 0.0f;
    
    for (int kh = 0; kh < kernel_h; kh++) {
        for (int kw = 0; kw < kernel_w; kw++) {
            int ih = (oh + padding - kh) * stride + ow * stride + kw - padding;
            if (ih >= 0 && ih < in_height && kw >= 0 && kw < kernel_w) {
                int weight_idx = ((in_c_group * kernel_h + kh) * kernel_w + kw) * out_channel_groups + g;
                int input_idx = ((n * in_channels + in_c_group) * in_height + ih) * in_width + ow;
                sum += input[input_idx] * weight[weight_idx];
            }
        }
    }
    
    if (bias != nullptr) {
        sum += bias[oc];
    }
    
    output[idx] = sum;
}

torch::Tensor conv_transpose2d_cuda(
    torch::Tensor input, 
    torch::Tensor weight, 
    torch::Tensor bias,
    int stride, int padding, int output_padding,
    int in_channels, int out_channels,
    int in_height, int in_width,
    int out_height, int out_width,
    int kernel_h, int kernel_w
) {
    auto batch_size = input.size(0);
    auto out_channels = weight.size(0);
    auto out_height_out = out_height;
    auto out_width_out = out_width;
    
    auto output = torch::zeros({batch_size, out_channels, out_height_out, out_width_out}, input.options());
    
    int total_elements = batch_size * out_channels * out_height_out * out_width_out;
    const int block_size = 256;
    const int num_blocks = (total_elements + block_size - 1) / block_size;
    
    const float* bias_ptr = bias.defined() ? bias.data_ptr<float>() : nullptr;
    
    conv_transpose2d_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias_ptr,
        output.data_ptr<float>(),
        batch_size, in_channels, out_channels,
        in_height, in_width,
        out_height_out, out_width_out,
        kernel_h, kernel_w,
        stride, padding, output_padding,
        1, 1
    );
    
    return output;
}
"""

conv_transpose2d_cpp_source = """
torch::Tensor conv_transpose2d_cuda(
    torch::Tensor input, 
    torch::Tensor weight, 
    torch::Tensor bias,
    int stride, int padding, int output_padding,
    int in_channels, int out_channels,
    int in_height, int in_width,
    int out_height, int out_width,
    int kernel_h, int kernel_w
);
"""

conv_transpose2d = load_inline(
    name="conv_transpose2d",
    cpp_sources=conv_transpose2d_cpp_source,
    cuda_sources=conv_transpose2d_source,
    functions=["conv_transpose2d_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: int = 1, padding: int = 0, output_padding: int = 0, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        self.bias = bias
        
        # Create weight and bias parameters
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels // groups, kernel_size[0], kernel_size[1]))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_buffer('bias', torch.tensor([]))
        
        self.conv_transpose2d = conv_transpose2d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, in_channels, in_height, in_width = x.shape
        out_height = (in_height - 1) * self.stride - 2 * self.padding + self.kernel_size[0] + self.output_padding
        out_width = (in_width - 1) * self.stride - 2 * self.padding + self.kernel_size[1] + self.output_padding
        
        return self.conv_transpose2d.conv_transpose2d_cuda(
            x, 
            self.weight, 
            self.bias if self.bias is not None else torch.tensor([]),
            self.stride, 
            self.padding, 
            self.output_padding,
            self.in_channels,
            self.out_channels,
            in_height,
            in_width,
            out_height,
            out_width,
            self.kernel_size[0],
            self.kernel_size[1]
        )