import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

depthwise_conv2d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void depthwise_conv2d_kernel(const float* input, const float* kernel, float* output,
    int batch_size, int in_channels, int in_height, int in_width,
    int kernel_h, int kernel_w, int stride_h, int stride_w,
    int padding_h, int padding_w, int dilation_h, int dilation_w) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * in_channels * ((in_height + 2 * padding_h - dilation_h * (kernel_h - 1) - 1) / stride_h + 1) * ((in_width + 2 * padding_w - dilation_w * (kernel_w - 1) - 1) / stride_w + 1);
    
    if (idx >= total_elements) return;
    
    int out_width = (in_width + 2 * padding_w - dilation_w * (kernel_w - 1) - 1) / stride_w + 1;
    int out_height = (in_height + 2 * padding_h - dilation_h * (kernel_h - 1) - 1) / stride_h + 1;
    
    int n = idx / (in_channels * out_height * out_width);
    int c = (idx / (out_height * out_width)) % in_channels;
    int oidx = idx % (out_height * out_width);
    int oh = oidx / out_width;
    int ow = oidx % out_width;
    
    int ih = oh * stride_h - padding_h;
    int iw = ow * stride_w - padding_w;
    
    float sum = 0;
    for (int kh = 0; kh < kernel_h; kh++) {
        for (int kw = 0; kw < kernel_w; kw++) {
            int ihi = ih + kh * dilation_h;
            int iwi = iw + kw * dilation_w;
            if (ihi >= 0 && ihi < in_height && iwi >= 0 && iwi < in_width) {
                sum += input[n * in_channels * in_height * in_width + c * in_height * in_width + ihi * in_width + iwi] * 
                       kernel[c * kernel_h * kernel_w + kh * kernel_w + kw];
            }
        }
    }
    output[idx] = sum;
}

torch::Tensor depthwise_conv2d_cuda(torch::Tensor input, torch::Tensor kernel, int stride_h, int stride_w, 
    int padding_h, int padding_w, int dilation_h, int dilation_w) {
    
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int in_height = input.size(2);
    int in_width = input.size(3);
    
    int out_height = (in_height + 2 * padding_h - dilation_h * (kernel_h - 1) - 1) / stride_h + 1;
    int out_width = (in_width + 2 * padding_w - dilation_w * (kernel_w - 1) - 1) / stride_w + 1;
    
    auto output = torch::zeros({batch_size, in_channels, out_height, out_width}, input.options());
    
    int block_size = 256;
    int total_elements = batch_size * in_channels * out_height * out_width;
    int num_blocks = (total_elements + block_size - 1) / block_size;
    
    depthwise_conv2d_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(), kernel.data_ptr<float>(), output.data_ptr<float>(),
        batch_size, in_channels, in_height, in_width,
        kernel_h, kernel_w, stride_h, stride_w,
        padding_h, padding_w, dilation_h, dilation_w);
    
    return output;
}
"""

depthwise_conv2d_cpp_source = """
torch::Tensor depthwise_conv2d_cuda(torch::Tensor input, torch::Tensor kernel, int stride_h, int stride_w, 
    int padding_h, int padding_w, int dilation_h, int dilation_w);
"""

depthwise_conv2d = load_inline(
    name="depthwise_conv2d",
    cpp_sources=depthwise_conv2d_cpp_source,
    cuda_sources=depthwise_conv2d_source,
    functions=["depthwise_conv2d_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size_h: int, kernel_size_w: int, stride_h: int = 1, stride_w: int = 1, padding_h: int = 0, padding_w: int = 0, dilation_h: int = 1, dilation_w: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.kernel_h = kernel_size_h
        self.kernel_w = kernel_size_w
        self.stride_h = stride_h
        self.stride_w = stride_w
        self.padding_h = padding_h
        self.padding_w = padding_w
        self.dilation_h = dilation_h
        self.dilation_w = dilation_w
        self.depthwise_conv2d = depthwise_conv2d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.depthwise_conv2d.depthwise_conv2d_cuda(
            x, 
            torch.ones(self.in_channels, self.kernel_h, self.kernel_w, device=x.device, dtype=x.dtype),
            self.stride_h, self.stride_w,
            self.padding_h, self.padding_w,
            self.dilation_h, self.dilation_w
        )