import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

conv_transpose3d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

template<typename scalar_t>
__global__ void conv_transpose3d_kernel(
    const scalar_t* input,
    const scalar_t* weight,
    const scalar_t* bias,
    scalar_t* output,
    int batch_size,
    int in_channels,
    int out_channels,
    int depth, int height, int width,
    int kernel_size,
    int stride,
    int padding,
    int output_padding,
    int in_depth, int in_height, int in_width,
    int out_depth, int out_height, int out_width
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * out_channels * out_depth * out_height * out_width;
    if (idx >= total_elements) return;

    int n = idx / (out_channels * out_depth * out_height * out_width);
    int c = (idx / (out_depth * out_height * out_width)) % out_channels;
    int d = (idx / (out_height * out_width)) % out_depth;
    int h = (idx / out_width) % out_height;
    int w = idx % out_width;

    scalar_t sum = 0;
    for (int kd = 0; kd < kernel_size; ++kd) {
        for (int kh = 0; kh < kernel_size; ++kh) {
            for (int kw = 0; kw < kernel_size; ++kw) {
                int d_in = d - kd * stride + padding;
                int h_in = h - kh * stride + padding;
                int w_in = w - kw * stride + padding;
                
                if (d_in >= 0 && d_in < in_depth && h_in >= 0 && h_in < in_height && w_in >= 0 && w_in < in_width) {
                    for (int ic = 0; ic < in_channels; ++ic) {
                        int weight_idx = ((c * in_channels + ic) * kernel_size + kd) * kernel_size + kh * kernel_size + kw;
                        int in_idx = ((n * in_channels + ic) * in_depth + d_in) * in_height + h_in) * in_width + w_in;
                        sum += input[in_idx] * weight[weight_idx];
                    }
                }
            }
        }
    }
    
    if (bias != nullptr) {
        sum += bias[c];
    }
    
    output[idx] = sum;
}

template<typename scalar_t>
torch::Tensor conv_transpose3d_cuda(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    int stride,
    int padding,
    int output_padding
) {
    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto in_depth = input.size(2);
    auto in_height = input.size(3);
    auto in_width = input.size(4);
    
    auto out_channels = weight.size(0);
    auto kernel_size = weight.size(2);
    
    auto out_depth = (in_depth - 1) * stride - 2 * padding + kernel_size + output_padding;
    auto out_height = (in_height - 1) * stride - 2 * padding + kernel_size + output_padding;
    auto out_width = (in_width - 1) * stride - 2 * padding + kernel_size + output_padding;
    
    auto output = torch::zeros({batch_size, out_channels, out_depth, out_height, out_width}, input.options());
    
    const int block_size = 256;
    const int total_elements = batch_size * out_channels * out_depth * out_height * out_width;
    const int num_blocks = (total_elements + block_size - 1) / block_size;
    
    const scalar_t* bias_ptr = bias.defined() ? bias.data_ptr<scalar_t>() : nullptr;
    
    conv_transpose3d_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<scalar_t>(),
        weight.data_ptr<scalar_t>(),
        bias_ptr,
        output.data_ptr<scalar_t>(),
        batch_size,
        in_channels,
        out_channels,
        in_depth, in_height, in_width,
        out_depth, out_height, out_width,
        kernel_size,
        stride,
        padding,
        output_padding,
        in_depth, in_height, in_width,
        out_depth, out_height, out_width
    );
    
    return output;
}

torch::Tensor conv_transpose3d_cuda_wrapper(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, int stride, int padding, int output_padding) {
    if (input.scalar_type() == torch::kFloat32) {
        return conv_transpose3d_cuda<float>(input, weight, bias, stride, padding, output_padding);
    } else if (input.scalar_type() == torch::kFloat64) {
        return conv_transpose3d_cuda<double>(input, weight, bias, stride, padding, output_padding);
    }
    return conv_transpose3d_cuda<float>(input, weight, bias, stride, padding, output_padding);
}
"""

conv_transpose3d_cpp_source = """
torch::Tensor conv_transpose3d_cuda_wrapper(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, int stride, int padding, int output_padding);
"""

conv_transpose3d = load_inline(
    name="conv_transpose3d",
    cpp_sources=conv_transpose3d_cpp_source,
    cuda_sources=conv_transpose3d_source,
    functions=["conv_transpose3d_cuda_wrapper"],
    verbose=True
)

class ModelNew(nn.Module):
    """
    Performs a transposed 3D convolution with square input and square kernel.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (int): Size of the square convolution kernel.
        stride (int, optional): Stride of the convolution. Defaults to 1.
        padding (int, optional): Padding applied to the input. Defaults to 0.
        output_padding (int, optional): Additional size added to one side of the output shape. Defaults to 0.
        groups (int, optional): Number of blocked connections from input channels to output channels. Defaults to 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, output_padding: int = 0, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        
        # Create weight and bias tensors
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels // groups, kernel_size, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.bias = None
        
        self.conv_transpose3d = conv_transpose3d
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the transposed 3D convolution.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, depth, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, depth_out, height_out, width_out).
        """
        return self.conv_transpose3d.conv_transpose3d_cuda_wrapper(x, self.weight, self.bias if self.bias is not None else torch.tensor(0.0, device=x.device), self.stride, self.padding, self.output_padding)