import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

avg_pool3d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void avg_pool3d_kernel(const float* input, float* output, int batch_size, int channels, int depth, int height, int width, int kernel_size, int stride, int padding) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_output_elements = batch_size * channels * ((depth + 2 * padding - kernel_size) / stride + 1) * ((height + 2 * padding - kernel_size) / stride + 1) * ((width + 2 * padding - kernel_size) / stride + 1);
    
    if (idx >= total_output_elements) return;
    
    int out_w = idx % ((width + 2 * padding - kernel_size) / stride + 1);
    int remainder = idx / ((width + 2 * padding - kernel_size) / stride + 1);
    int out_h = remainder % ((height + 2 * padding - kernel_size) / stride + 1);
    remainder = remainder / ((height + 2 * padding - kernel_size) / stride + 1);
    int out_d = remainder % ((depth + 2 * padding - kernel_size) / stride + 1);
    remainder = remainder / ((depth + 2 * padding - kernel_size) / stride + 1);
    int b = remainder / channels;
    int c = remainder % channels;
    
    int in_d = out_d * stride - padding;
    int in_h = out_h * stride - padding;
    int in_w = out_w * stride - padding;
    
    float sum = 0.0f;
    int count = 0;
    for (int kd = 0; kd < kernel_size; ++kd) {
        for (int kh = 0; kh < kernel_size; ++kh) {
            for (int kw = 0; kw < kernel_size; ++kw) {
                int d = in_d + kd;
                int h = in_h + kh;
                int w = in_w + kw;
                if (d >= 0 && d < depth && h >= 0 && h < height && w >= 0 && w < width) {
                    sum += input[((b * channels + c) * depth + d) * height * width + h * width + w];
                    count++;
                }
            }
        }
    }
    output[idx] = sum / count;
}

torch::Tensor avg_pool3d_cuda(torch::Tensor input, int kernel_size, int stride, int padding) {
    int batch_size = input.size(0);
    int channels = input.size(1);
    int depth = input.size(2);
    int height = input.size(3);
    int width = input.size(4);
    
    int out_depth = (depth + 2 * padding - kernel_size) / stride + 1;
    int out_height = (height + 2 * padding - kernel_size) / stride + 1;
    int out_width = (width + 2 * padding - kernel_size) / stride + 1;
    
    auto output = torch::zeros({batch_size, channels, out_depth, out_height, out_width}, input.options());
    
    int total_elements = batch_size * channels * out_depth * out_height * out_width;
    const int block_size = 256;
    const int num_blocks = (total_elements + block_size - 1) / block_size;
    
    avg_pool3d_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size, channels, depth, height, width,
        kernel_size, stride, padding
    );
    
    return output;
}
"""

avg_pool3d_cpp_source = "torch::Tensor avg_pool3d_cuda(torch::Tensor input, int kernel_size, int stride, int padding);"

avg_pool3d = load_inline(
    name="avg_pool3d",
    cpp_sources=avg_pool3d_cpp_source,
    cuda_sources=avg_pool3d_source,
    functions=["avg_pool3d_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0):
        super(ModelNew, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding
        self.avg_pool3d = avg_pool3d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.avg_pool3d.avg_pool3d_cuda(x, self.kernel_size, self.stride, self.padding)