import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = """#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void avg_pool2d_kernel(const float* input, float* output,
                                   int batch_size, int channels, int height, int width,
                                   int kernel_size, int stride, int padding) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * channels * (height * stride - 1) * (width * stride - 1);
    if (idx < total) {
        int c = (idx / (height * stride * width)) % channels;
        int h = (idx / (stride * width)) % height;
        int w = idx % (stride * width);
        
        float sum = 0.0f;
        for (int h0 = h - padding; h0 <= h; h0++) {
            for (int w0 = w - padding; w0 <= w; w0++) {
                int input_idx = c * (height * width) + h0 * width + w0;
                sum += input[input_idx];
            }
        }
        output[idx] = sum / kernel_size;
    }
}

torch::Tensor avg_pool2d_cuda(torch::Tensor input, int64_t kernel_size, int64_t stride, int64_t padding) {
    int batch_size = input.size(0);
    int channels = input.size(1);
    int height = input.size(2);
    int width = input.size(3);
    
    int output_h = (height - 1) * stride - 1;
    int output_w = (width - 1) * stride - 1;
    int total = batch_size * channels * output_h * output_w;
    
    auto output = torch::empty({batch_size, channels, output_h, output_w}, input.options());
    
    const int block_size = 256;
    const int num_blocks = (total + block_size - 1) / block_size;
    
    avg_pool2d_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size, channels, height, width,
        kernel_size, stride, padding,
        0,
        1
    );
    
    return output;
}"""

cpp_source = """torch::Tensor avg_pool2d_cuda(torch::Tensor input, int64_t kernel_size, int64_t stride, int64_t padding);"""

custom_ops = load_inline(
    name="custom_ops",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['avg_pool2d_cuda'],
    verbose=False,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"],
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x):
        return custom_ops.avg_pool2d_cuda(x)
