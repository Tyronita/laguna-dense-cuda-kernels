import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

conv_transpose2d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

template<typename T>
__global__ void conv_transpose2d_kernel(
    const T* input,
    const T* weight,
    const T* bias,
    T* output,
    int batch_size,
    int in_channels,
    int out_channels,
    int in_height,
    int in_width,
    int out_height,
    int out_width,
    int kernel_h,
    int kernel_w,
    int stride_h,
    int stride_w,
    int padding_h,
    int padding_w,
    int output_padding_h,
    int output_padding_w,
    int dilation_h,
    int dilation_w,
    int groups
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * out_channels * out_height * out_width;
    
    if (idx >= total_elements) return;
    
    int n = idx / (out_channels * out_height * out_width);
    int remainder = idx % (out_channels * out_height * out_width);
    int c = remainder / (out_height * out_width);
    remainder = remainder % (out_height * out_width);
    int h = remainder / out_width;
    int w = remainder % out_width;
    
    T sum = bias != nullptr ? bias[c] : T(0);
    
    int k_h_start = (h < padding_h) ? 0 : (h - padding_h + dilation_h - 1) / dilation_h;
    int k_h_end = min((h * stride_h + kernel_h - padding_h) / dilation_h, kernel_h);
    int k_w_start = (w < padding_w) ? 0 : (w - padding_w + dilation_w - 1) / dilation_w;
    int k_w_end = min((w * stride_w + kernel_w - padding_w) / dilation_w, kernel_w);
    
    for (int kh = k_h_start; kh < k_h_end; kh++) {
        for (int kw = k_w_start; kw < k_w_end; kw++) {
            int in_h = h * stride_h + kh * dilation_h - padding_h;
            int in_w = w * stride_w + kw * dilation_w - padding_w;
            
            if (in_h >= 0 && in_h < in_height && in_w >= 0 && in_w < in_width) {
                int in_c = c % groups;
                int group = c / groups;
                int weight_idx = ((group * in_channels + in_c) * kernel_h + kh) * kernel_w + kw;
                int in_idx = ((n * in_channels + in_c) * in_height + in_h) * in_width + in_w;
                sum += input[in_idx] * weight[weight_idx];
            }
        }
    }
    
    output[idx] = sum;
}

torch::Tensor conv_transpose2d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int stride_h, int stride_w,
    int padding_h, int padding_w,
    int output_padding_h, int output_padding_w,
    int dilation_h, int dilation_w
) {
    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto in_height = input.size(2);
    auto in_width = input.size(3);
    
    auto out_channels = weight.size(0);
    auto kernel_h = weight.size(2);
    auto kernel_w = weight.size(3);
    
    auto out_height = (in_height - 1) * stride_h - 2 * padding_h + dilation_h * kernel_h + output_padding_h;
    auto out_width = (in_width - 1) * stride_w - 2 * padding_w + dilation_w * kernel_w + output_padding_w;
    
    auto output = torch::zeros({batch_size, out_channels, out_height, out_width}, input.options());
    
    const int block_size = 256;
    const int num_blocks = (batch_size * out_channels * out_height * out_width + block_size - 1) / block_size;
    
    if (num_blocks > 0) {
        conv_transpose2d_kernel<<<num_blocks, block_size>>>(
            input.data_ptr<float>(),
            weight.data_ptr<float>(),
            bias.defined() ? bias.data_ptr<float>() : nullptr,
            output.data_ptr<float>(),
            batch_size, in_channels, out_channels,
            in_height, in_width, out_height, out_width,
            kernel_h, kernel_w,
            stride_h, stride_w,
            padding_h, padding_w,
            output_padding_h, output_padding_w,
            dilation_h, dilation_w,
            1
        );
    }
    
    return output;
}
"""

conv_transpose2d_cpp_source = """
torch::Tensor conv_transpose2d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int stride_h, int stride_w,
    int padding_h, int padding_w,
    int output_padding_h, int output_padding_w,
    int dilation_h, int dilation_w
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
    """
    Performs a transposed 2D convolution operation with asymmetric input and kernel size.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (tuple): Tuple of integers representing the kernel size (height, width).
        stride (tuple, optional): Tuple of integers representing the stride of the convolution. Defaults to (1, 1).
        padding (tuple, optional): Tuple of integers representing the padding applied to the input. Defaults to (0, 0).
        output_padding (tuple, optional): Tuple of integers representing the additional size added to one side of the output shape. Defaults to (0, 0).
        dilation (tuple, optional): Tuple of integers representing the spacing between kernel elements. Defaults to (1, 1).
        groups (int, optional): Number of blocked connections from input channels to output channels. Defaults to 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1), padding: tuple = (0, 0), output_padding: tuple = (0, 0), dilation: tuple = (1, 1), groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        
        # Create weight and bias parameters
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels // groups, kernel_size[0], kernel_size[1]))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_buffer('bias', None)
        
        self.conv_transpose2d = conv_transpose2d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the transposed 2D convolution.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height_in, width_in).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height_out, width_out).
        """
        return self.conv_transpose2d.conv_transpose2d_cuda(
            x,
            self.weight,
            self.bias,
            self.stride[0], self.stride[1],
            self.padding[0], self.padding[1],
            self.output_padding[0], self.output_padding[1],
            self.dilation[0], self.dilation[1]
        )