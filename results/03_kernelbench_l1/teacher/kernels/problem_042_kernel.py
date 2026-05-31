import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

maxpool2d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void maxpool2d_kernel(const float* input, float* output, 
                                  int batch_size, int channels, int in_height, int in_width,
                                  int out_height, int out_width, int kernel_size, int stride, int padding) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * channels * out_height * out_width;
    
    if (idx < total_elements) {
        int n = idx / (channels * out_height * out_width);
        int c = (idx / (out_height * out_width)) % channels;
        int oh = (idx / out_width) % out_height;
        int ow = idx % out_width;
        
        int ih_start = oh * stride - padding;
        int iw_start = ow * stride - padding;
        
        float max_val = -INFINITY;
        for (int kh = 0; kh < kernel_size; kh++) {
            for (int kw = 0; kw < kernel_size; kw++) {
                int ih = ih_start + kh;
                int iw = iw_start + kw;
                if (ih >= 0 && ih < in_height && iw >= 0 && iw < in_width) {
                    int input_idx = ((n * channels + c) * in_height + ih) * in_width + iw;
                    max_val = fmaxf(max_val, input[input_idx]);
                }
            }
        }
        int output_idx = ((n * channels + c) * out_height + oh) * out_width + ow;
        output[output_idx] = max_val;
    }
}

torch::Tensor maxpool2d_cuda(torch::Tensor input, int kernel_size, int stride, int padding, int dilation) {
    auto batch_size = input.size(0);
    auto channels = input.size(1);
    auto in_height = input.size(2);
    auto in_width = input.size(3);
    
    auto out_height = (in_height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    auto out_width = (in_width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    
    auto output = torch::zeros({batch_size, channels, out_height, out_width}, input.options());
    
    const int block_size = 256;
    const int num_blocks = (batch_size * channels * out_height * out_width + block_size - 1) / block_size;
    
    maxpool2d_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(), output.data_ptr<float>(),
        batch_size, channels, in_height, in_width,
        out_height, out_width, kernel_size, stride, padding
    );
    
    return output;
}
"""

maxpool2d_cpp_source = "torch::Tensor maxpool2d_cuda(torch::Tensor input, int kernel_size, int stride, int padding, int dilation);"

maxpool2d = load_inline(
    name="maxpool2d",
    cpp_sources=maxpool2d_cpp_source,
    cuda_sources=maxpool2d_source,
    functions=["maxpool2d_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    """
    Simple model that performs Max Pooling 2D with custom CUDA implementation.
    """
    def __init__(self, kernel_size: int, stride: int, padding: int, dilation: int):
        """
        Initializes the Max Pooling 2D layer.

        Args:
            kernel_size (int): Size of the pooling window.
            stride (int): Stride of the pooling window.
            padding (int): Padding to be applied before pooling.
            dilation (int): Spacing between kernel elements.
        """
        super(ModelNew, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.maxpool2d_cuda = maxpool2d.maxpool2d_cuda

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Max Pooling 2D to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: Output tensor after Max Pooling 2D, shape (batch_size, channels, pooled_height, pooled_width).
        """
        return self.maxpool2d_cuda(x, self.kernel_size, self.stride, self.padding, self.dilation)