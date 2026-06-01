import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Depthwise Conv2D CUDA Source
depthwise_conv2d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void depthwise_conv2d_kernel(const float* input, const float* weight, const float* bias, float* output,
                                         int batch_size, int in_channels, int in_height, int in_width,
                                         int out_height, int out_width, int kernel_size, int stride, int padding, int dilation) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * out_height * out_width * in_channels;
    if (idx >= total_elements) return;

    int n = idx / (out_height * out_width * in_channels);
    int remainder = idx % (out_height * out_width * in_channels);
    int c = remainder / (out_height * out_width);
    remainder = remainder % (out_height * out_width);
    int h_out = remainder / out_width;
    int w_out = remainder % out_width;

    int h_in_start = h_out * stride - padding;
    int w_in_start = w_out * stride - padding;

    float sum = bias != nullptr ? bias[c] : 0.0f;
    for (int kh = 0; kh < kernel_size; kh++) {
        for (int kw = 0; kw < kernel_size; kw++) {
            int h_in = h_in_start + kh * dilation;
            int w_in = w_in_start + kw * dilation;
            if (h_in >= 0 && h_in < in_height && w_in >= 0 && w_in < in_width) {
                sum += input[n * in_channels * in_height * in_width + c * in_height * in_width + h_in * in_width + w_in] * 
                       weight[c * kernel_size * kernel_size + kh * kernel_size + kw];
            }
        }
    }
    output[idx] = sum;
}

torch::Tensor depthwise_conv2d_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, int stride, int padding, int dilation) {
    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto in_height = input.size(2);
    auto in_width = input.size(3);
    auto kernel_size = weight.size(2);
    
    auto out_height = (in_height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    auto out_width = (in_width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    
    auto output = torch::zeros({batch_size, in_channels, out_height, out_width}, input.options());
    
    const int block_size = 256;
    const int num_blocks = (batch_size * out_height * out_width * in_channels + block_size - 1) / block_size;
    
    const float* bias_ptr = bias.defined() ? bias.data_ptr<float>() : nullptr;
    depthwise_conv2d_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), bias_ptr, output.data_ptr<float>(),
        batch_size, in_channels, in_height, in_width, out_height, out_width, kernel_size, stride, padding, dilation);
    
    return output;
}
"""

depthwise_conv2d_cpp = "torch::Tensor depthwise_conv2d_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, int stride, int padding, int dilation);"

# Pointwise Conv2D CUDA Source
pointwise_conv2d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void pointwise_conv2d_kernel(const float* input, const float* weight, const float* bias, float* output,
                                         int batch_size, int in_channels, int in_height, int in_width,
                                         int out_channels, int out_height, int out_width) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * out_height * out_width * out_channels;
    if (idx >= total_elements) return;

    int n = idx / (out_height * out_width * out_channels);
    int remainder = idx % (out_height * out_width * out_channels);
    int c_out = remainder / (out_height * out_width);
    remainder = remainder % (out_height * out_width);
    int h = remainder / out_width;
    int w = remainder % out_width;

    float sum = bias.defined() ? bias[c_out] : 0.0f;
    for (int c_in = 0; c_in < in_channels; c_in++) {
        sum += input[n * in_channels * in_height * in_width + c_in * in_height * in_width + h * in_width + w] * 
               weight[c_out * in_channels + c_in];
    }
    output[idx] = sum;
}

torch::Tensor pointwise_conv2d_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias) {
    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto in_height = input.size(2);
    auto in_width = input.size(3);
    auto out_channels = weight.size(0);
    auto out_height = in_height;
    auto out_width = in_width;
    
    auto output = torch::zeros({batch_size, out_channels, out_height, out_width}, input.options());
    
    const int block_size = 256;
    const int num_blocks = (batch_size * out_height * out_width * out_channels + block_size - 1) / block_size;
    
    const float* bias_ptr = bias.defined() ? bias.data_ptr<float>() : nullptr;
    pointwise_conv2d_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), bias_ptr, output.data_ptr<float>(),
        batch_size, in_channels, in_height, in_width, out_channels, out_height, out_width);
    
    return output;
}
"""

pointwise_conv2d_cpp = "torch::Tensor pointwise_conv2d_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias);"

# Load inline CUDA extensions
depthwise_conv2d = load_inline(
    name="depthwise_conv2d",
    cpp_sources=depthwise_conv2d_cpp,
    cuda_sources=depthwise_conv2d_source,
    functions=["depthwise_conv2d_cuda"],
    verbose=False
)

pointwise_conv2d = load_inline(
    name="pointwise_conv2d",
    cpp_sources=pointwise_conv2d_cpp,
    cuda_sources=pointwise_conv2d_source,
    functions=["pointwise_conv2d_cuda"],
    verbose=False
)

class ModelNew(nn.Module):
    """
    Performs a depthwise-separable 2D convolution operation using custom CUDA operators.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (int): Size of the convolution kernel.
        stride (int, optional): Stride of the convolution. Defaults to 1.
        padding (int, optional): Padding applied to the input. Defaults to 0.
        dilation (int, optional): Spacing between kernel elements. Defaults to 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.bias = bias
        
        # Initialize weights and bias
        self.weight_depthwise = nn.Parameter(torch.randn(in_channels, 1, kernel_size, kernel_size) * 0.01)
        self.weight_pointwise = nn.Parameter(torch.randn(out_channels, in_channels) * 0.01)
        if bias:
            self.bias_depthwise = nn.Parameter(torch.zeros(in_channels))
            self.bias_pointwise = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_buffer('bias_depthwise', None)
            self.register_buffer('bias_pointwise', None)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the depthwise-separable 2D convolution.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height_out, width_out).
        """
        # Depthwise convolution
        bias_depthwise = self.bias_depthwise if self.bias else None
        x = depthwise_conv2d.depthwise_conv2d_cuda(
            x, self.weight_depthwise, bias_depthwise, self.stride, self.padding, self.dilation
        )
        
        # Pointwise convolution
        bias_pointwise = self.bias_pointwise if self.bias else None
        x = pointwise_conv2d.pointwise_conv2d_cuda(x, self.weight_pointwise, bias_pointwise)
        
        return x