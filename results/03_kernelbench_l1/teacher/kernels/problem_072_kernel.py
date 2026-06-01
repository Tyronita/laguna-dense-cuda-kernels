import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

conv_transpose3d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void conv_transpose3d_kernel(
    const float* input, 
    const float* weight, 
    const float* bias,
    float* output,
    int batch_size, int in_channels, int out_channels,
    int in_depth, int in_height, int in_width,
    int out_depth, int out_height, int out_width,
    int kernel_d, int kernel_h, int kernel_w,
    int stride_d, int stride_h, int stride_w,
    int pad_d, int pad_h, int pad_w,
    int out_pad_d, int out_pad_h, int out_pad_w,
    int groups
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * out_channels * out_depth * out_height * out_width;
    
    if (idx >= total_elements) return;
    
    int temp = idx;
    int od = temp % out_depth; temp /= out_depth;
    int oh = temp % out_height; temp /= out_height;
    int ow = temp % out_width; temp /= out_width;
    int oc = temp % out_channels; temp /= out_channels;
    int n = temp;
    
    int in_c = oc % in_channels;
    int g = oc / (in_channels / groups);
    
    int d_in_start = (od - out_pad_d + pad_d) * stride_d;
    int d_in_end = min(d_in_start + kernel_d, in_depth + pad_d);
    d_in_start = max(d_in_start, 0);
    
    int h_in_start = (oh - out_pad_h + pad_h) * stride_h;
    int h_in_end = min(h_in_start + kernel_h, in_height + pad_h);
    h_in_start = max(h_in_start, 0);
    
    int w_in_start = (ow - out_pad_w + pad_w) * stride_w;
    int w_in_end = min(w_in_start + kernel_w, in_width + pad_w);
    w_in_start = max(w_in_start, 0);
    
    float sum = 0.0f;
    
    for (int kd = d_in_start; kd < d_in_end; ++kd) {
        for (int kh = h_in_start; kh < h_in_end; ++kh) {
            for (int kw = w_in_start; kw < w_in_end; ++kw) {
                int k_d = kd - (od - out_pad_d + pad_d) * stride_d;
                int k_h = kh - (oh - out_pad_h + pad_h) * stride_h;
                int k_w = kw - (ow - out_pad_w + pad_w) * stride_w;
                
                if (k_d >= 0 && k_d < kernel_d && k_h >= 0 && k_h < kernel_h && k_w >= 0 && k_w < kernel_w) {
                    int weight_idx = ((oc / (in_channels / groups)) * in_channels + in_c) * kernel_d * kernel_h * kernel_w + 
                                     ((kd * kernel_h + kh) * kernel_w + kw);
                    int input_idx = ((n * in_channels + in_c) * in_depth + kd) * in_height + kh) * in_width + kw;
                    sum += weight[weight_idx] * input[input_idx];
                }
            }
        }
    }
    
    if (bias != nullptr) {
        sum += bias[oc];
    }
    
    output[idx] = sum;
}

torch::Tensor conv_transpose3d_cuda(
    torch::Tensor input, 
    torch::Tensor weight, 
    torch::Tensor bias,
    int stride_d, int stride_h, int stride_w,
    int pad_d, int pad_h, int pad_w,
    int out_pad_d, int out_pad_h, int out_pad_w
) {
    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto in_depth = input.size(2);
    auto in_height = input.size(3);
    auto in_width = input.size(4);
    
    auto kernel_d = weight.size(2);
    auto kernel_h = weight.size(3);
    auto kernel_w = weight.size(4);
    
    auto out_depth = (in_depth - 1) * stride_d - 2 * pad_d + kernel_d + out_pad_d;
    auto out_height = (in_height - 1) * stride_h - 2 * pad_h + kernel_h + out_pad_h;
    auto out_width = (in_width - 1) * stride_w - 2 * pad_w + kernel_w + out_pad_w;
    
    auto out_channels = weight.size(0);
    
    auto output = torch::zeros({batch_size, out_channels, out_depth, out_height, out_width}, input.options());
    
    const int block_size = 256;
    const int num_blocks = (output.numel() + block_size - 1) / block_size;
    
    const float* bias_ptr = bias.defined() ? bias.data_ptr<float>() : nullptr;
    
    conv_transpose3d_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias_ptr,
        output.data_ptr<float>(),
        batch_size, in_channels, out_channels,
        in_depth, in_height, in_width,
        out_depth, out_height, out_width,
        kernel_d, kernel_h, kernel_w,
        stride_d, stride_h, stride_w,
        pad_d, pad_h, pad_w,
        out_pad_d, out_pad_h, out_pad_w,
        1
    );
    
    return output;
}
"""

conv_transpose3d_cpp_source = """
torch::Tensor conv_transpose3d_cuda(
    torch::Tensor input, 
    torch::Tensor weight, 
    torch::Tensor bias,
    int stride_d, int stride_h, int stride_w,
    int pad_d, int pad_h, int pad_w,
    int out_pad_d, int out_pad_h, int out_pad_w
);
"""

conv_transpose3d = load_inline(
    name="conv_transpose3d",
    cpp_sources=conv_transpose3d_cpp_source,
    cuda_sources=conv_transpose3d_source,
    functions=["conv_transpose3d_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1, 1), padding: tuple = (0, 0, 0), output_padding: tuple = (0, 0, 0), groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels // groups, *kernel_size))
        self.bias = nn.Parameter(torch.randn(out_channels)) if bias else None
        
        self.conv_transpose3d = conv_transpose3d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv_transpose3d.conv_transpose3d_cuda(
            x, 
            self.weight, 
            self.bias if self.bias is not None else torch.Tensor(),
            self.stride[0], self.stride[1], self.stride[2],
            self.padding[0], self.padding[1], self.padding[2],
            self.output_padding[0], self.output_padding[1], self.output_padding[2]
        )