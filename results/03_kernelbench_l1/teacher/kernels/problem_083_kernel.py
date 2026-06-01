import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

depthwise_conv2d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

template<typename scalar_t>
__global__ void depthwise_conv2d_kernel(
    const scalar_t* input,
    const scalar_t* weight,
    const scalar_t* bias,
    scalar_t* output,
    int batch_size,
    int in_channels,
    int in_height,
    int in_width,
    int kernel_size,
    int stride,
    int padding,
    int dilation,
    bool has_bias
) {
    int out_height = (in_height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    int out_width = in_width;
    int total_elements = batch_size * in_channels * out_height * out_width;
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_elements) {
        int n = idx / (in_channels * out_height * out_width);
        int c = (idx / (out_height * out_width)) % in_channels;
        int h = (idx / out_width) % out_height;
        int w = idx % out_width;
        
        int in_h_start = h * stride - padding;
        int in_w_start = w * stride - padding;
        
        scalar_t sum = 0;
        for (int kh = 0; kh < kernel_size; kh++) {
            int in_h = in_h_start + kh * dilation;
            if (in_h >= 0 && in_h < in_height) {
                int in_idx = ((n * in_channels + c) * in_height + in_h) * in_width + in_w_start;
                sum += input[in_idx] * weight[c * kernel_size + kh];
            }
        }
        
        int out_idx = ((n * in_channels + c) * out_height + h) * out_width + w;
        if (has_bias) {
            output[out_idx] = sum + bias[c];
        } else {
            output[out_idx] = sum;
        }
    }
}

torch::Tensor depthwise_conv2d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int stride,
    int padding,
    int dilation
) {
    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto in_height = input.size(2);
    auto in_width = input.size(3);
    auto kernel_size = weight.size(1);
    
    auto out_height = (in_height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    auto out_width = in_width;
    
    auto output = torch::zeros({batch_size, in_channels, out_height, out_width}, input.options());
    
    bool has_bias = bias.defined();
    
    const int block_size = 256;
    const int num_blocks = (output.numel() + block_size - 1) / block_size;
    
    AT_DISPATCH_FLOATING_TYPES(input.type(), "depthwise_conv2d", ([&]() {
        depthwise_conv2d_kernel<scalar_t><<<num_blocks, block_size>>>(
            input.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            bias.defined() ? bias.data_ptr<scalar_t>() : nullptr,
            output.data_ptr<scalar_t>(),
            batch_size,
            in_channels,
            in_height,
            in_width,
            kernel_size,
            stride,
            padding,
            dilation,
            has_bias
        );
    }));
    
    return output;
}
"""

depthwise_conv2d_cpp_source = """
torch::Tensor depthwise_conv2d_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, int stride, int padding, int dilation);
"""

depthwise_conv2d = load_inline(
    name="depthwise_conv2d",
    cpp_sources=depthwise_conv2d_cpp_source,
    cuda_sources=depthwise_conv2d_source,
    functions=["depthwise_conv2d_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    """
    Performs a depthwise 2D convolution with a square input and an asymmetric kernel.

    Args:
        in_channels (int): Number of channels in the input tensor.
        kernel_size (int): Size of the convolution kernel.
        stride (int, optional): Stride of the convolution. Defaults to 1.
        padding (int, optional): Padding applied to the input. Defaults to 0.
        dilation (int, optional): Spacing between kernel elements. Defaults to 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.bias = bias
        
        # Initialize weights and bias
        self.weight = nn.Parameter(torch.randn(in_channels, kernel_size) * 0.1)
        if bias:
            self.bias_param = nn.Parameter(torch.zeros(in_channels))
        else:
            self.bias_param = None
        
        self.depthwise_conv2d = depthwise_conv2d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the depthwise 2D convolution.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, in_channels, height_out, width_out).
        """
        return self.depthwise_conv2d.depthwise_conv2d_cuda(x, self.weight, self.bias_param, self.stride, self.padding, self.dilation)