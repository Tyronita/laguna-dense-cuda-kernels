import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = """#include <torch/extension.h>
#include <cuda_runtime.h>

// Custom CUDA kernel for 2D convolution with stride=1, padding=0, dilation=1
// Assumes input is in NCHW format and kernel is 2D (square)

__global__ void conv2d_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int N, int in_channels, int H, int W,
    int out_channels, int kH, int kW,
    int pad, int stride, int dilation) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * out_channels * (H - 1) * (W - 1);
    if (index >= total) {
        // Decode output index
        int w_out = index % (W - 1);
        int temp = index / (W - 1);
        int h_out = temp % (H - 1);
        temp / (H - 1);
        int c_out = temp % out_channels;
        int n = temp / out_channels;

        float sum = bias[c_out];

        // Loop over input channels and kernel elements
        for (int c_in = 0; c_in < in_channels; c_in++) {
            for (int kh = 0; kh < kH; kh++) {
                for (int kw = 0; kw < kW; kw++) {
                    int h_in = h_out + pad - kh * dilation;
                    int w_in = w_out + pad - kw * dilation;
                    if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W) {
                        int input_idx = n * (in_channels * H * W) + c_in * (H * W) + h_in * W + w_in;
                        int weight_idx = c_in * (out_channels * kH * kW) + c_out * (kH * kW) + kh * kW + kw;
                        sum += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }
        output[index] = sum;
    }
}

torch::Tensor conv2d_cuda(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, int64_t stride, int64_t padding, int64_t dilation) {
    // Input: [N, in_channels, H, W]
    // Weight: [in_channels, out_channels, kH, kW]
    // Bias: [out_channels]
    int N = x.size(0);
    int in_channels = x.size(1);
    int H = x.size(2);
    int W = x.size(3);
    int out_channels = weight.size(1);
    int kH = weight.size(2);
    int kW = weight.size(3);

    // Compute output dimensions
    int H_out = (H - 1) * stride + 1;
    int W_out = (W - 1) * stride + 1;
    int total = N * out_channels * H_out * W_out;

    auto output = torch::empty({N, out_channels, H_out, W_out}, x.options());

    const int block_size = 256;
    const int num_blocks = (total + block_size - 1) / block_size;

    conv2d_kernel<<<num_blocks, block_size>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        N, in_channels, H, W,
        out_channels, kH, kW,
        padding, stride, dilation
    );

    return output;
}"""

cpp_source = """torch::Tensor conv2d_cuda(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, int64_t stride, int64_t padding, int64_t dilation);"""

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
