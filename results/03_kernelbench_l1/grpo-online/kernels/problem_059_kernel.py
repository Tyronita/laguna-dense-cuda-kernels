import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = """#include <torch/extension.h>
#include <cuda_runtime.h>

// Custom CUDA kernel for 3D convolution with stride=1, padding=0, dilation=1
// Assumes input tensor shape: [batch_size, in_channels, height, width, depth]
// and output tensor shape: [batch_size, out_channels, height_out, width_out, depth_out]

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
    int kernel_size,
    int stride,
    int padding,
    int dilation) {

    // Compute output dimensions
    int h_out = height - 2 * padding + kernel_size - 1;
    int w_out = width - 2 * padding + kernel_size - 1;
    int d_out = depth - 2 * padding + kernel_size - 1;

    // Compute output indices
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * out_channels * h_out * w_out * d_out;
    if (idx < total) {
        // Decode output index into (n, oc, h, w, d)
        int d = idx % d_out;
        int temp = idx / d_out;
        int w = temp % w_out;
        temp /= w_out;
        int h = temp % h_out;
        temp /= h_out;
        int oc = temp % out_channels;
        int n = temp / out_channels;

        float sum = bias[oc];

        // Convolution loop
        for (int c = 0; c < in_channels; c++) {
            for (int kh = 0; kh < kernel_size; kh++) {
                for (int kw = 0; kw < kernel_size; kw++) {
                    int h_in = h + padding - kh;
                    int w_in = w + padding - kw;
                    int d_in = d + padding - kh;
                    if (h_in >= 0 && h_in < height && w_in >= 0 && w_in < width && d_in >= 0 && d_in < depth) {
                        int input_idx = n * (in_channels * height * width * depth) + c * (height * width * depth) + h_in * (width * depth) + w_in * depth + d_in;
                        int weight_idx = oc * (in_channels * kernel_size * kernel_size) + c * (kernel_size * kernel_size) + kh * kernel_size + kw;
                        sum += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }
        output[idx] = sum;
    }
}

// The forward function exposed via PyBind11
torch::Tensor forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    int64_t stride,
    int64_t padding,
    int64_t dilation) {

    // Input tensor of shape [batch_size, in_channels, height, width, depth]
    // Output tensor of shape [batch_size, out_channels, height_out, width_out, depth_out]
    int batch_size = x.size(0);
    int in_channels = x.size(1);
    int height = x.size(2);
    int width = x.size(3);
    int depth = x.size(4);

    int out_channels = weight.size(0);
    int kernel_size = weight.size(2);
    int kH = weight.size(3);
    int kW = weight.size(4);

    // Compute output dimensions
    int h_out = (height - 2 * padding + kernel_size - 1) / stride;
    int w_out = (width - 2 * padding + kernel_size - 1) / stride;
    int d_out = (depth - 2 * padding + kernel_size - 1) / stride;

    auto output = torch::empty({batch_size, out_channels, h_out, w_out, d_out}, x.options());

    int total = batch_size * out_channels * h_out * w_out * d_out;
    int block_size = 256;
    int num_blocks = (total + block_size - 1) / block_size;

    conv3d_kernel<<<num_blocks, block_size>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        height,
        width,
        depth,
        out_channels,
        kernel_size,
        stride,
        padding,
        dilation
    );

    return output;
}"""

cpp_source = """torch::Tensor forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    int64_t stride,
    int64_t padding,
    int64_t dilation);"""

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
