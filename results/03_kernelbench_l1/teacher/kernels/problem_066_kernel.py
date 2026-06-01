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
                              int stride_d, int stride_h, int stride_w,
                              int padding_d, int padding_h, int padding_w,
                              int dilation_d, int dilation_h, int dilation_w,
                              int out_depth, int out_height, int out_width) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * out_channels * out_depth * out_height * out_width;
    
    if (idx >= total_elements) return;
    
    // Decode index
    int ow = idx % out_width;
    int oh = (idx / out_width) % out_height;
    int od = (idx / (out_width * out_height)) % out_depth;
    int oc = (idx / (out_width * out_height * out_depth)) % out_channels;
    int ob = idx / (out_width * out_height * out_depth * out_channels);
    
    float sum = 0.0f;
    
    for (int ic = 0; ic < in_channels; ic++) {
        for (int kd = 0; kd < kernel_d; kd++) {
            int d_in = od * stride_d + kd * dilation_d - padding_d;
            if (d_in < 0 || d_in >= depth) continue;
            
            for (int kh = 0; kh < kernel_h; kh++) {
                int h_in = oh * stride_h + kh * dilation_h - padding_h;
                if (h_in < 0 || h_in >= height) continue;
                
                for (int kw = 0; kw < kernel_w; kw++) {
                    int w_in = ow * stride_w + kw * dilation_w - padding_w;
                    if (w_in < 0 || w_in >= width) continue;
                    
                    int weight_idx = ((oc * in_channels + ic) * kernel_d + kd) * kernel_h + kh) * kernel_w + kw;
                    int input_idx = ((ob * in_channels + ic) * depth + d_in) * height + h_in) * width + w_in;
                    
                    sum += weight[weight_idx] * input[input_idx];
                }
            }
        }
    }
    
    if (bias != nullptr) {
        sum += bias[oc];
    }
    
    int out_idx = ((ob * out_channels + oc) * out_depth + od) * out_height + oh) * out_width + ow;
    output[out_idx] = sum;
}

torch::Tensor conv3d_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
                          int stride_d, int stride_h, int stride_w,
                          int padding_d, int padding_h, int padding_w,
                          int dilation_d, int dilation_h, int dilation_w) {
    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto depth = input.size(2);
    auto height = input.size(3);
    auto width = input.size(4);
    
    auto out_channels = weight.size(0);
    auto kernel_d = weight.size(2);
    auto kernel_h = weight.size(3);
    auto kernel_w = weight.size(4);
    
    auto out_depth = (depth + 2 * padding_d - dilation_d * (kernel_d - 1) - 1) / stride_d + 1;
    auto out_height = (height + 2 * padding_h - dilation_h * (kernel_h - 1) - 1) / stride_h + 1;
    auto out_width = (width + 2 * padding_w - dilation_w * (kernel_w - 1) - 1) / stride_w + 1;
    
    auto output = torch::zeros({batch_size, out_channels, out_depth, out_height, out_width}, input.options());
    
    const int block_size = 256;
    const int total_elements = batch_size * out_channels * out_depth * out_height * out_width;
    const int num_blocks = (total_elements + block_size - 1) / block_size;
    
    const float* bias_ptr = bias.defined() ? bias.data_ptr<float>() : nullptr;
    
    conv3d_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), bias_ptr, output.data_ptr<float>(),
        batch_size, in_channels, out_channels, depth, height, width,
        kernel_d, kernel_h, kernel_w, stride_d, stride_h, stride_w,
        padding_d, padding_h, padding_w, dilation_d, dilation_h, dilation_w,
        out_depth, out_height, out_width
    );
    
    return output;
}
"""

conv3d_cpp_source = """
torch::Tensor conv3d_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
                          int stride_d, int stride_h, int stride_w,
                          int padding_d, int padding_h, int padding_w,
                          int dilation_d, int dilation_h, int dilation_w);
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
    Performs a standard 3D convolution operation with asymmetric input and kernel sizes.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (tuple): Size of the convolution kernel in the form (kernel_size_d, kernel_size_h, kernel_size_w).
        stride (tuple, optional): Stride of the convolution in the form (stride_d, stride_h, stride_w). Defaults to (1, 1, 1).
        padding (tuple, optional): Padding applied to the input in the form (padding_d, padding_h, padding_w). Defaults to (0, 0, 0).
        dilation (tuple, optional): Spacing between kernel elements in the form (dilation_d, dilation_h, dilation_w). Defaults to (1, 1, 1).
        groups (int, optional): Number of blocked connections from input channels to output channels. Defaults to 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1, 1), padding: tuple = (0, 0, 0), dilation: tuple = (1, 1, 1), groups: int = 1, bias: bool = False):
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
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels // groups, *kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_buffer('bias', None)
        
        self.conv3d = conv3d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 3D convolution.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, depth, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, depth_out, height_out, width_out).
        """
        return self.conv3d.conv3d_cuda(
            x, self.weight, self.bias if self.bias is not None else torch.tensor([]),
            self.stride[0], self.stride[1], self.stride[2],
            self.padding[0], self.padding[1], self.padding[2],
            self.dilation[0], self.dilation[1], self.dilation[2]
        )