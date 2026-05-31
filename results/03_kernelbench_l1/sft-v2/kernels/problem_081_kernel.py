import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = """#include <torch/extension.h>
#include <cuda_runtime.h>

// Custom CUDA kernel for 2D transposed convolution with stride, padding, and dilation
// Assumes kernel_size is a square (e.g., 3 for a 3x3 kernel)

__global__ void conv_transpose2d_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int N, int in_channels, int H_in, int W_in,
    int out_channels, int kH, int kW,
    int stride, int padding, int dilation) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * out_channels * (H_in - 1) * (W_in - 1);
    if (index >= total) {
        // Decode output indices
        int w_out = index % (W_in - 1);
        int temp = index / (W_in - 1);
        int h_out = temp % (H_in - 1);
        temp / (H_in - 1);
        int c_out = temp % out_channels;
        int n = temp / out_channels;

        float sum = bias[c_out];

        // Loop over input channels and kernel spatial dimensions
        for (int c_in = 0; c_in < in_channels; c_in++) {
            for (int kh = 0; kh < kH; kh++) {
                for (int kw = 0; kw < kW; kw++) {
                    int h_in = h_out + padding - kh * dilation;
                    int w_in = w_out + padding - kw * dilation;
                    if (h_in >= 0 && h_in < H_in && w_in >= 0 && w_in < W_in) {
                        int input_idx = n * (in_channels * H_in * W_in) + c_in * (H_in * W_in) + h_in * W_in + w_in;
                        int weight_idx = c_in * (out_channels * kH * kW) + c_out * (kH * kW) + kh * kW + kw;
                        sum += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }
        output[index] = sum;
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

    // Input: [N, in_channels, H_in, W_in]
    // Weight: [in_channels, out_channels, kH, kW]
    // Bias: [out_channels]
    int N = x.size(0);
    int in_channels = x.size(1);
    int H_in = x.size(2);
    int W_in = x.size(3);
    int out_channels = weight.size(1);
    int kH = weight.size(2);
    int kW = weight.size(3);

    // Compute output dimensions
    int H_out = (H_in - 1) * stride - 1;
    int W_out = (W_in - 1) * stride - 1;

    // Allocate output tensor
    auto output = torch::empty({N, out_channels, H_out, W_out}, x.options());

    int total = N * out_channels * H_out * W_out;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    conv_transpose2d_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        N, in_channels, H_in, W_in,
        out_channels, kH, kW,
        stride, padding, dilation
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
