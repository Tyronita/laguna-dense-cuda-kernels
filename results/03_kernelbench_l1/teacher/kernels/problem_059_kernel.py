import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

conv3d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void conv3d_kernel(const float* input, const float* weight, const float* bias, float* output,
                               int batch_size, int in_channels, int out_channels,
                               int in_depth, int in_height, int in_width,
                               int kernel_size, int stride, int padding, int dilation, int groups) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * out_channels * ((in_height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1) * 
                         ((in_width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1) * 
                         ((in_depth + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1);
    
    if (idx >= total_elements) return;
    
    // Compute output indices
    int d_out = idx / (out_channels * ((in_height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1) * ((in_width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1));
    int remainder = idx % (out_channels * ((in_height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1) * ((in_width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1));
    int h_out = remainder / (out_channels * ((in_width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1));
    remainder = remainder % (out_channels * ((in_width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1));
    int w_out = remainder / out_channels;
    int n_out = remainder % out_channels;
    
    int h_in = h_out * stride - padding;
    int w_in = w_in * stride - padding;
    int d_in = d_out * stride - padding;
    
    float sum = 0;
    for (int c_in = 0; c_in < in_channels; ++c_in) {
        for (int kh = 0; kh < kernel_size; ++kh) {
            for (int kw = 0; kw < kernel_size; ++kw) {
                int h_in_k = h_in + kh * dilation;
                int w_in_k = w_in + kw * dilation;
                int d_in_k = d_in;
                
                if (h_in_k >= 0 && h_in_k < in_height && w_in_k >= 0 && w_in_k < in_width && d_in_k >= 0 && d_in_k < in_depth) {
                    int weight_idx = ((c_in / groups) * kernel_size * kernel_size + kh * kernel_size + kw) * out_channels + n_out;
                    int input_idx = ((n_out / groups) * in_channels + c_in) * in_depth * in_height * in_width + d_in_k * in_height * in_width + h_in_k * in_width + w_in_k;
                    sum += input[input_idx] * weight[weight_idx];
                }
            }
        }
    }
    
    if (bias) {
        int bias_idx = n_out;
        output[idx] = sum + bias[bias_idx];
    } else {
        output[idx] = sum;
    }
}

torch::Tensor conv3d_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, 
                          int stride, int padding, int dilation, int groups) {
    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto in_depth = input.size(2);
    auto in_height = input.size(3);
    auto in_width = input.size(4);
    
    auto out_channels = weight.size(0);
    auto kernel_size = weight.size(2);
    
    auto out_depth = (in_depth + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    auto out_height = (in_height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    auto out_width = (in_width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    
    auto output = torch::zeros({batch_size, out_channels, out_depth, out_height, out_width}, input.options());
    
    const int block_size = 256;
    const int total_elements = batch_size * out_channels * out_depth * out_height * out_width;
    const int num_blocks = (total_elements + block_size - 1) / block_size;
    
    conv3d_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), 
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        batch_size, in_channels, out_channels,
        in_depth, in_height, in_width,
        kernel_size, stride, padding, dilation, groups
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
    """
    Performs a standard 3D convolution operation with an asymmetric input and a square kernel.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (int): Size of the square convolution kernel (kernel_size x kernel_size).
        stride (int, optional): Stride of the convolution. Defaults to 1.
        padding (int, optional): Padding applied to the input. Defaults to 0.
        dilation (int, optional): Spacing between kernel elements. Defaults to 1.
        groups (int, optional): Number of blocked connections from input channels to output channels. Defaults to 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
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
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels // groups, kernel_size, kernel_size, 1))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_buffer('bias', None)
        
        self.conv3d = conv3d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 3D convolution.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width, depth).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height_out, width_out, depth_out).
        """
        return self.conv3d.conv3d_cuda(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)