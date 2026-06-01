import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

conv2d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void conv2d_kernel(const float* input, const float* weight, const float* bias, float* output,
                              int batch_size, int in_channels, int out_channels, int in_height, int in_width,
                              int kernel_h, int kernel_w, int stride, int padding, int dilation, int groups) {
    int out_h = (in_height + 2 * padding - dilation * (kernel_h - 1) - 1) / stride + 1;
    int out_w = (in_width + 2 * padding - dilation * (kernel_w - 1) - 1) / stride + 1;
    
    int total_elements = batch_size * out_channels * out_h * out_w;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < total_elements) {
        int n = idx / (out_channels * out_h * out_w);
        int c = (idx % (out_channels * out_h * out_w)) / (out_h * out_w);
        int h = (idx % (out_h * out_w)) / out_w;
        int w = idx % out_w;
        
        int group = c / (out_channels / groups);
        int in_c = c % (out_channels / groups) + group * (in_channels / groups);
        
        float sum = 0.0f;
        for (int kh = 0; kh < kernel_h; kh++) {
            for (int kw = 0; kw < kernel_w; kw++) {
                int in_h = h * stride - padding + kh * dilation;
                int in_w = w * stride - padding + kw * dilation;
                if (in_h >= 0 && in_h < in_height && in_w >= 0 && in_w < in_width) {
                    int weight_idx = ((c / (out_channels / groups)) * (in_channels / groups) + in_c) * kernel_h * kernel_w + kh * kernel_w + kw;
                    int input_idx = n * in_channels * in_height * in_width + in_c * in_height * in_width + in_h * in_width + in_w;
                    sum += input[input_idx] * weight[weight_idx];
                }
            }
        }
        
        int out_idx = n * out_channels * out_h * out_w + c * out_h * out_w + h * out_w + w;
        output[out_idx] = sum;
    }
}

torch::Tensor conv2d_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, 
                          int stride, int padding, int dilation, int groups) {
    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto in_height = input.size(2);
    auto in_width = input.size(3);
    auto out_channels = weight.size(0);
    auto kernel_h = weight.size(2);
    auto kernel_w = weight.size(3);
    
    int out_h = (in_height + 2 * padding - dilation * (kernel_h - 1) - 1) / stride + 1;
    int out_w = (in_width + 2 * padding - dilation * (kernel_w - 1) - 1) / stride + 1;
    
    auto output = torch::zeros({batch_size, out_channels, out_h, out_w}, input.options());
    
    const int block_size = 256;
    int total_elements = batch_size * out_channels * out_h * out_w;
    const int num_blocks = (total_elements + block_size - 1) / block_size;
    
    conv2d_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), 
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(), batch_size, in_channels, out_channels, 
        in_height, in_width, kernel_h, kernel_w, stride, padding, dilation, groups
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
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding if isinstance(padding, int) else padding[0]
        self.dilation = dilation if isinstance(dilation, int) else dilation[0]
        self.groups = groups
        
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels // groups, kernel_size[0], kernel_size[1]))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_buffer('bias', None)
        
        self.conv2d = conv2d
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv2d.conv2d_cuda(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)