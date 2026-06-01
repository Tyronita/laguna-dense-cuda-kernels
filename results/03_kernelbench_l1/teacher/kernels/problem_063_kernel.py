import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

conv2d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

template<typename scalar_t>
__global__ void conv2d_kernel(
    const scalar_t* input,
    const scalar_t* weight,
    const scalar_t* bias,
    scalar_t* output,
    int batch_size, int in_channels, int out_channels,
    int in_height, int in_width,
    int out_height, int out_width,
    int kernel_size, int stride, int padding, int dilation, int groups
) {
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_outputs = batch_size * out_channels * out_height * out_width;
    
    if (out_idx >= total_outputs) return;
    
    int n = out_idx / (out_channels * out_height * out_width);
    int oc = (out_idx / (out_height * out_width)) % out_channels;
    int oh = (out_idx / out_width) % out_height;
    int ow = out_idx % out_width;
    
    int ic = oc % groups == 0 ? oc / groups : oc;
    int g = oc / groups;
    
    scalar_t sum = bias != nullptr ? bias[oc] : 0;
    
    for (int kh = 0; kh < kernel_size; kh++) {
        for (int kw = 0; kw < kernel_size; kw++) {
            int ih = oh * stride - padding + kh * dilation;
            int iw = ow * stride - padding + kw * dilation;
            
            if (ih >= 0 && ih < in_height && iw >= 0 && iw < in_width) {
                int weight_idx = ((kh * kernel_size + kw) * in_channels + ic) * (out_channels / groups) + g;
                int input_idx = ((n * in_channels + ic) * in_height + ih) * in_width + iw;
                sum += input[input_idx] * weight[weight_idx];
            }
        }
    }
    
    output[out_idx] = sum;
}

torch::Tensor conv2d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int stride, int padding, int dilation, int groups
) {
    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto in_height = input.size(2);
    auto in_width = input.size(3);
    
    auto out_channels = weight.size(0);
    auto kernel_size = weight.size(2);
    
    auto out_height = (in_height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    auto out_width = (in_width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    
    auto output = torch::zeros({batch_size, out_channels, out_height, out_width}, input.options());
    
    const int block_size = 256;
    const int total_outputs = batch_size * out_channels * out_height * out_width;
    const int num_blocks = (total_outputs + block_size - 1) / block_size;
    
    conv2d_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        batch_size, in_channels, out_channels,
        in_height, in_width,
        out_height, out_width,
        kernel_size, stride, padding, dilation, groups
    );
    
    return output;
}
"""

conv2d_cpp_source = "torch::Tensor conv2d_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, int stride, int padding, int dilation, int groups);"

conv2d = load_inline(
    name="conv2d",
    cpp_sources=conv2d_cpp_source,
    cuda_sources=conv2d_source,
    functions=["conv2d_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
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
        
        # Initialize weights and bias
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels // groups, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_buffer('bias', None)
        
        self.conv2d = conv2d
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv2d.conv2d_cuda(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)