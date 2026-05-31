import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = """#include <torch/extension.h>
#include <cuda_runtime.h>

// Custom CUDA kernel for 2D convolution with stride=1, padding=0, dilation=1
// Assumes input tensor x of shape (N, C_in, H, W) and output tensor y of shape (N, C_out, H_out, W)
// Convolution weights are of shape (C_out, C_in, kH, kW)

__global__ void conv2d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int N, int C_in, int H, int W,
    int C_out, int kH, int kW,
    int H_out, int W_out) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C_out * H_out * W_out;
    if (idx < total) {
        // Decode linear index into (n, oc, h_out, w_out)
        int w_out = idx % W_out;
        int temp = idx / W_out;
        int h_out = temp % H_out;
        temp / H_out;
        int oc = temp % C_out;
        int n = temp / C_out;

        float sum = bias[oc];

        // Loop over input channels and kernel spatial dimensions
        for (int c_in = 0; c_in < C_in; c_in++) {
            for (int kh = 0; kh < kH; kh++) {
                for (int kw = 0; kw < kW; kw++) {
                    int h_in = h_out + kh - 1;  // padding = 0, dilation = 1
                    int w_in = w_out + kw - 1;
                    if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W) {
                        int input_idx = n * (C_in * H * W) + c_in * (H * W) + h_in * W + w_in;
                        int weight_idx = oc * (C_in * kH * kW) + c_in * (kH * kW) + kh * kW + kw;
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

    // Input: x of shape [N, C_in, H, W]
    // Weight: weight of shape [C_out, C_in, kH, kW]
    // Bias: bias of shape [C_out]
    int N = x.size(0);
    int C_in = x.size(1);
    int H = x.size(2);
    int W = x.size(3);
    int C_out = weight.size(0);
    int kH = weight.size(2);
    int kW = weight.size(3);

    // Compute output dimensions (assuming stride=1, padding=0, dilation=1)
    int H_out = H - kH + 1;
    int W_out = W - kW + 1;

    auto output = torch::empty({N, C_out, H_out, W_out}, x.options());

    int total = N * C_out * H_out * W_out;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    conv2d_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C_in, H, W,
        C_out, kH, kW,
        H_out, W_out
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
