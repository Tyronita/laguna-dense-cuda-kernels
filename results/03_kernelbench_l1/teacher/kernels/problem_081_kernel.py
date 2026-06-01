import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

conv_transpose2d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void conv_transpose2d_kernel(
    const float* input, 
    const float* weight, 
    const float* bias,
    float* output,
    int batch_size, int in_channels, int in_height, int in_width,
    int out_channels, int out_height, int out_width,
    int kernel_size, int stride, int padding, int dilation
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

    float sum = bias != nullptr ? bias[c] : 0.0f;

    for (int kh = 0; kh < kernel_size; kh++) {
        for (int kw = 0; kw < kernel_size; kw++) {
            int h_in = h * stride - padding + kh * dilation;
            int w_in = w * stride - padding + kw * dilation;
            
            if (h_in >= 0 && h_in < in_height && w_in >= 0 && w_in < in_width) {
                for (int ic = 0; ic < in_channels; ic++) {
                    int weight_idx = ((kh * kernel_size + kw) * in_channels + ic) * out_channels + c;
                    int input_idx = ((n * in_channels + ic) * in_height + h_in) * in_width + w_in;
                    sum += weight[weight_idx] * input[input_idx];
                }
            }
        }
    }

    int out_idx = ((n * out_channels + c) * out_height + h) * out_width + w;
    output[out_idx] = sum;
}

torch::Tensor conv_transpose2d_cuda(
    torch::Tensor input, 
    torch::Tensor weight, 
    torch::Tensor bias,
    int stride, int padding, int dilation
) {
    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto in_height = input.size(2);
    auto in_width = input.size(3);
    
    auto out_channels = weight.size(0);
    auto kernel_size = weight.size(2);
    
    auto out_height = (in_height - 1) * stride + kernel_size - 2 * padding;
    auto out_width = (in_width - 1) * stride + kernel_size - 2 * padding;

    auto output = torch::zeros({batch_size, out_channels, out_height, out_width}, input.options());
    
    const int block_size = 256;
    const int num_blocks = (batch_size * out_channels * out_height * out_width + block_size - 1) / block_size;
    
    conv_transpose2d_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        batch_size, in_channels, in_height, in_width,
        out_channels, out_height, out_width,
        kernel_size, stride, padding, dilation
    );
    
    return output;
}
"""

conv_transpose2d_cpp_source = """
torch::Tensor conv_transpose2d_cuda(
    torch::Tensor input, 
    torch::Tensor weight, 
    torch::Tensor bias,
    int stride, int padding, int dilation
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
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.bias = bias
        
        # Create weight and bias tensors
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        if bias:
            self.bias_tensor = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_buffer('bias_tensor', None)
        
        self.conv_transpose2d = conv_transpose2d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias_tensor = self.bias_tensor if self.bias else torch.Tensor()
        return self.conv_transpose2d.conv_transpose2d_cuda(x, self.weight, bias_tensor, self.stride, self.padding, self.dilation)