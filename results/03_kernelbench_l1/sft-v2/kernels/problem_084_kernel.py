import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = """#include <torch/extension.h>
#include <cuda_runtime.h>

// Custom CUDA kernel for 2D convolution with stride and padding
__global__ void conv2d_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int batch_size,
    int in_channels,
    int height_in,
    int width_in,
    int out_channels,
    int kernel_size,
    int stride,
    int padding) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * out_channels * (height_in - 1) * (width_in - 1);
    if (idx < total) {
        // Decode output indices
        int h_out = (height_in - 1) * (width_in - 1);
        int w_out = height_in * width_in;
        int c_out = (h_out / w_out) % out_channels;
        int b = idx / (w_out * out_channels);
        int h = h_out / w_out;
        int w = idx % w_out;

        float sum = bias[c_out];
        for (int c_in = 0; c_in < in_channels; c_in++) {
            for (int kh = 0; kh < kernel_size; kh++) {
                for (int kw = 0; kw < kernel_size; kw++) {
                    int h_in = h - padding + kh;
                    int w_in = w - padding + kw;
                    if (h_in >= 0 && h_in < height_in && w_in >= 0 && w_in < width_in) {
                        int input_idx = b * (in_channels * height_in * width_in) + 
                                       c_in * (height_in * width_in) + 
                                       h_in * width_in + w_in;
                        int weight_idx = c_in * (out_channels * kernel_size * kernel_size) + 
                                       c_out * (kernel_size * kernel_size) + 
                                       kh * kernel_size + kw;
                        sum += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }
        output[idx] = sum;
    }
}

torch::Tensor conv2d_cuda(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, int64_t stride, int64_t padding) {
    auto batch_size = x.size(0);
    auto in_channels = x.size(1);
    auto height_in = x.size(2);
    auto width_in = x.size(3);
    auto out_channels = weight.size(0);
    auto height_out = (height_in - 1) * (width_in - 1);
    auto width_out = height_in * width_in;

    auto output = torch::empty({batch_size, out_channels, height_out, width_out}, x.options());
    int total = batch_size * out_channels * height_out * width_out;
    const int block_size = 256;
    const int num_blocks = (total + block_size - 1) / block_size;

    conv2d_kernel<<<num_blocks, block_size>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        height_in,
        width_in,
        out_channels,
        weight.size(2),
        stride,
        padding
    );

    return output;
}"""

cpp_source = """torch::Tensor conv2d_cuda(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, int64_t stride, int64_t padding);"""

custom_ops = load_inline(
    name="custom_ops",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['conv2d_cuda'],
    verbose=False,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"],
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x):
        return custom_ops.conv2d_cuda(x)
