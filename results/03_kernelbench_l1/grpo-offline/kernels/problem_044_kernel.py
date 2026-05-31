import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = """#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void avg_pool1d_kernel(
    const float* input,
    float* output,
    int batch_size,
    int in_channels,
    int input_length,
    int kernel_size,
    int stride,
    int padding) {

    int out_channels = in_channels;
    int out_h = (input_length - padding + kernel_size) / stride;
    int out_w = 1;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size * out_channels * out_h * out_w) {
        int b = idx / (out_channels * out_h * out_w);
        int c = (idx / (out_h * out_w)) % out_channels;
        int h = (idx / out_w) % out_h;

        float sum = 0.0f;
        int in_h_start = h * stride - padding;
        int in_w_start = 0;

        for (int i = in_h_start; i < input_length; i += kernel_size) {
            for (int j = in_w_start; j < input_length; j += kernel_size) {
                int in_idx = b * (in_channels * input_length) + c * (input_length) + i * input_length + j;
                sum += input[in_idx] / kernel_size;
            }
        }
        output[idx] = sum / kernel_size;
    }
}

torch::Tensor avg_pool1d_cuda(
    torch::Tensor x,
    int64_t kernel_size,
    int64_t stride,
    int64_t padding) {
    
    auto batch_size = x.size(0);
    auto in_channels = x.size(1);
    auto input_length = x.size(2);

    int out_channels = in_channels;
    int out_h = (input_length - padding + kernel_size) / stride;
    int out_w = 1;

    auto output = torch::empty({batch_size, out_channels, out_h, out_w}, x.options());

    const int block_size = 256;
    const int num_blocks = batch_size * out_channels * out_h * out_w;

    avg_pool1d_kernel<<<num_blocks, block_size>>>(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        input_length,
        kernel_size,
        stride,
        padding
    );

    return output;
}"""

cpp_source = """torch::Tensor avg_pool1d_cuda(
    torch::Tensor x,
    int64_t kernel_size,
    int64_t stride,
    int64_t padding);"""

custom_ops = load_inline(
    name="custom_ops",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['avg_pool1d_cuda'],
    verbose=False,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"],
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x):
        return custom_ops.avg_pool1d_cuda(x)
