import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

conv2d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void conv2d_kernel(const float* input, const float* weight, const float* bias, float* output,
                              int batch_size, int in_channels, int out_channels, int in_height, int in_width,
                              int out_height, int out_width, int kernel_h, int kernel_w,
                              int stride_h, int stride_w, int pad_h, int pad_w, int dil_h, int dil_w) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * out_channels * out_height * out_width;
    if (idx >= total_elements) return;

    int n = idx / (out_channels * out_height * out_width);
    int c = (idx / (out_height * out_width)) % out_channels;
    int h = (idx / out_width) % out_height;
    int w = idx % out_width;

    float sum = 0.0f;
    for (int kh = 0; kh < kernel_h; kh++) {
        for (int kw = 0; kw < kernel_w; kw++) {
            int in_h = h * stride_h - pad_h + kh * dil_h;
            int in_w = w * stride_w - pad_w + kw * dil_w;
            if (in_h >= 0 && in_h < in_height && in_w >= 0 && in_w < in_width) {
                sum += input[n * in_channels * in_height * in_width + c * in_height * in_width + in_h * in_width + in_w] * 
                       weight[c * in_channels * kernel_h * kernel_w + kh * kernel_w + kw];
            }
        }
    }
    if (bias != nullptr) {
        sum += bias[c];
    }
    output[idx] = sum;
}

torch::Tensor conv2d_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
                          int stride_h, int stride_w, int pad_h, int pad_w, int dil_h, int dil_w) {
    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto in_height = input.size(2);
    auto in_width = input.size(3);
    auto out_channels = weight.size(0);
    auto kernel_h = weight.size(2);
    auto kernel_w = weight.size(3);

    auto out_height = (in_height + 2 * pad_h - dil_h * (kernel_h - 1) - 1) / stride_h + 1;
    auto out_width = (in_width + 2 * pad_w - dil_w * (kernel_w - 1) - 1) / stride_w + 1;

    auto output = torch::zeros({batch_size, out_channels, out_height, out_width}, input.options());

    const int block_size = 256;
    const int total_elements = batch_size * out_channels * out_height * out_width;
    const int num_blocks = (total_elements + block_size - 1) / block_size;

    const float* bias_ptr = bias.defined() ? bias.data_ptr<float>() : nullptr;

    conv2d_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), bias_ptr, output.data_ptr<float>(),
        batch_size, in_channels, out_channels, in_height, in_width,
        out_height, out_width, kernel_h, kernel_w,
        stride_h, stride_w, pad_h, pad_w, dil_h, dil_w
    );

    return output;
}
"""

conv2d_cpp_source = """
torch::Tensor conv2d_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
                          int stride_h, int stride_w, int pad_h, int pad_w, int dil_h, int dil_w);
"""

conv2d = load_inline(
    name="conv2d",
    cpp_sources=conv2d_cpp_source,
    cuda_sources=conv2d_source,
    functions=["conv2d_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    """
    Performs a standard 2D convolution operation with square input and asymmetric kernel, with dilation and padding.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (tuple): Size of the convolution kernel (height, width). 
        stride (int, optional): Stride of the convolution. Defaults to 1.
        padding (tuple, optional): Padding applied to the input (top/bottom, left/right). Defaults to (0, 0).
        dilation (tuple, optional): Spacing between kernel elements (height, width). Defaults to (1, 1).
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: int = 1, padding: tuple = (0, 0), dilation: tuple = (1, 1), bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.bias = bias
        
        # Create weight and bias parameters
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size[0], kernel_size[1]))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_buffer('bias', torch.tensor([]))
        
        # Load the custom CUDA operator
        self.conv2d = conv2d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 2D convolution.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height_out, width_out).
        """
        return self.conv2d.conv2d_cuda(
            x, self.weight, self.bias,
            self.stride, self.stride,
            self.padding[0], self.padding[1],
            self.dilation[0], self.dilation[1]
        )