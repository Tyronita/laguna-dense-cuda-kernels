import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = """#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Custom CUDA kernel for 3D convolution with stride=1, padding=0, dilation=1
// Assumes input shape: [batch_size, in_channels, height, width, depth]
// and output shape: [batch_size, out_channels, height_out, width_out, depth_out]
// with stride=1 and padding=0, dilation=1.

__global__ void conv3d_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int batch_size,
    int in_channels,
    int height,
    int width,
    int depth,
    int out_channels,
    int kernel_h,
    int kernel_w,
    int stride,
    int padding) {

    // Compute output dimensions
    int h_out = height - 1;
    int w_out = width - 1;
    int d_out = depth - 1;

    // Compute output indices
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * out_channels * h_out * w_out * d_out;
    if (index >= total) return;

    // Decode output index
    int d_out_idx = index % d_out;
    int temp = index / d_out;
    int w_out_idx = temp % w_out;
    temp /= w_out;
    int h_out_idx = temp % h_out;
    int c_out = temp / h_out;
    int b = c_out / out_channels;
    c_out % out_channels = c_out % out_channels;

    float sum = bias[c_out];

    // Convolution loop
    for (int c_in = 0; c_in < in_channels; c_in++) {
        for (int kh = 0; kh < kernel_h; kh++) {
            for (int kw = 0; kw < kernel_w; kw++) {
                int h_in = h_out_idx + padding - kh;
                int w_in = w_out_idx + padding - kw;
                if (h_in >= 0 && h_in < height && w_in >= 0 && w_in < width) {
                    int input_idx = b * (in_channels * height * width * depth) +
                                  c_in * (height * width * depth) +
                                  h_in * (width * depth) +
                                  w_in * depth +
                                  d_out_idx;
                    int weight_idx = c_in * (out_channels * kernel_h * kernel_w) +
                                   c_out * (kernel_h * kernel_w) +
                                   kh * kernel_w +
                                   kw;
                    sum += input[input_idx] * weight[weight_idx];
                }
            }
        }
    }

    output[index] = sum;
}

// The forward function exposed via PyBind11.
torch::Tensor forward(
    torch::Tensor x,
    torch::Tensor conv3d_weight,
    torch::Tensor conv3d_bias) {

    // Input dimensions
    int batch_size = x.size(0);
    int in_channels = x.size(1);
    int height = x.size(2);
    int width = x.size(3);
    int depth = x.size(4);

    int out_channels = conv3d_weight.size(0);
    int kernel_h = conv3d_weight.size(2);
    int kernel_w = conv3d_weight.size(3);

    // Compute output dimensions
    int h_out = height - 1;
    int w_out = width - 1;
    int d_out = depth - 1;

    // Allocate output tensor
    auto output = torch::empty({batch_size, out_channels, h_out, w_out, d_out}, x.options());

    int total = batch_size * out_channels * h_out * w_out * d_out;
    int block_size = 256;
    int num_blocks = (total + block_size - 1) / block_size;

    conv3d_kernel<<<num_blocks, block_size>>>(
        x.data_ptr<float>(),
        conv3d_weight.data_ptr<float>(),
        conv3d_bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        height,
        width,
        depth,
        out_channels,
        kernel_h,
        kernel_w,
        1,
        0
    );

    return output;
}"""

cpp_source = """torch::Tensor forward(
    torch::Tensor x,
    torch::Tensor conv3d_weight,
    torch::Tensor conv3d_bias);"""

custom_ops = load_inline(
    name="custom_ops",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['forward'],
    verbose=False,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"],
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x):
        return custom_ops.forward(x)
