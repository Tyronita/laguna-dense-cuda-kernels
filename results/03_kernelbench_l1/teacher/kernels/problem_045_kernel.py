import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

avg_pool2d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void avg_pool2d_kernel(const float* input, float* output, 
                                   int batch_size, int channels, int in_height, int in_width,
                                   int out_height, int out_width, int kernel_size, int stride, int padding) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * channels * out_height * out_width;
    
    if (idx < total_elements) {
        int n = idx / (channels * out_height * out_width);
        int c = (idx / (out_height * out_width)) % channels;
        int oh = (idx / (out_width)) % out_height;
        int ow = idx % out_width;
        
        int ih_start = oh * stride - padding;
        int iw_start = ow * stride - padding;
        
        float sum = 0.0f;
        int count = 0;
        
        for (int kh = 0; kh < kernel_size; kh++) {
            for (int kw = 0; kw < kernel_size; kw++) {
                int ih = ih_start + kh;
                int iw = iw_start + kw;
                
                if (ih >= 0 && ih < in_height && iw >= 0 && iw < in_width) {
                    sum += input[n * channels * in_height * in_width + 
                                c * in_height * in_width + 
                                ih * in_width + iw];
                    count++;
                }
            }
        }
        
        output[idx] = sum / count;
    }
}

torch::Tensor avg_pool2d_cuda(torch::Tensor input, int kernel_size, int stride, int padding) {
    auto batch_size = input.size(0);
    auto channels = input.size(1);
    auto in_height = input.size(2);
    auto in_width = input.size(3);
    
    auto out_height = (in_height + 2 * padding - kernel_size) / stride + 1;
    auto out_width = (in_width + 2 * padding - kernel_size) / stride + 1;
    
    auto output = torch::zeros({batch_size, channels, out_height, out_width}, input.options());
    
    const int block_size = 256;
    const int num_blocks = (batch_size * channels * out_height * out_width + block_size - 1) / block_size;
    
    avg_pool2d_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size, channels, in_height, in_width,
        out_height, out_width, kernel_size, stride, padding
    );
    
    return output;
}
"""

avg_pool2d_cpp_source = "torch::Tensor avg_pool2d_cuda(torch::Tensor input, int kernel_size, int stride, int padding);"

avg_pool2d = load_inline(
    name="avg_pool2d",
    cpp_sources=avg_pool2d_cpp_source,
    cuda_sources=avg_pool2d_source,
    functions=["avg_pool2d_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    """
    Simple model that performs 2D Average Pooling with custom CUDA implementation.
    """
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0):
        """
        Initializes the Average Pooling layer.

        Args:
            kernel_size (int): Size of the pooling window.
            stride (int, optional): Stride of the pooling operation. Defaults to None (same as kernel_size).
            padding (int, optional): Padding applied to the input tensor. Defaults to 0.
        """
        super(ModelNew, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding
        self.avg_pool2d = avg_pool2d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies 2D Average Pooling to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: Output tensor with Average Pooling applied.
        """
        return self.avg_pool2d.avg_pool2d_cuda(x, self.kernel_size, self.stride, self.padding)