import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = """#include <torch/extension.h>
#include <cuda_runtime.h>

// Custom CUDA kernel for 2D convolution with stride, padding, dilation
// Each block processes one output element corresponding to one (n, c) pair.
// The kernel computes the convolution sum over the input channels and the kernel spatial dimensions.

__global__ void conv2d_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int N, int in_channels, int H, int W,
    int out_channels, int kH, int kW,
    int stride_h, int stride_w, int pad_h, int pad_w, int dilation_h, int dilation_w) {

    // Each block corresponds to one (n, c) pair
    int n = blockIdx.x;
    int c = blockIdx.y;

    float sum = bias[c];

    // Loop over input channels
    for (int ic = 0; ic < in_channels; ic++) {
        // Compute the convolution window for this input channel
        int inH = (H - pad_h) * dilation_h + kH - pad_h;
        int inW = (W - pad_w) * dilation_w + kW - pad_w;

        // Loop over kernel height and width
        for (int kh = 0; kh < kH; kh++) {
            int inH_idx = inH + kh * stride_h;
            for (int kw = 0; kw < kW; kw++) {
                int inW_idx = inW + kw * stride_w;
                if (inH_idx >= 0 && inH_idx < H && inW_idx >= 0 && inW_idx < W) {
                    // Index for input tensor: (n, ic, inH_idx, inW_idx)
                    int input_idx = n * (in_channels * H * W) + ic * (H * W) + inH_idx * W + inW_idx;
                    // Index for weight tensor: (ic, c, kh, kw)
                    int weight_idx = ic * (out_channels * kH * kW) + c * (kH * kW) + kh * kW + kw;
                    
                    sum += input[input_idx] * weight[weight_idx];
                }
            }
        }
    }

    // Write the output tensor for this (n, c) pair
    int out_idx = n * (out_channels * H * W) + c * (H * W);
    output[out_idx] = sum;
}

// The forward function exposed via PyBind11.
// Assumes input is a CUDA tensor of shape [N, in_channels, H, W]
// and weight is a CUDA tensor of shape [in_channels, out_channels, kH, kW].
// Bias is a CUDA tensor of shape [out_channels].

torch::Tensor forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    int64_t stride_h,
    int64_t stride_w,
    int64_t pad_h,
    int64_t pad_w,
    int64_t dilation_h,
    int64_t dilation_w) {

    // x: [N, in_channels, H, W]
    // weight: [in_channels, out_channels, kH, kW]
    // bias: [out_channels]

    int N = x.size(0);
    int in_channels = x.size(1);
    int H = x.size(2);
    int W = x.size(3);
    int out_channels = weight.size(1);
    int kH = weight.size(2);
    int kW = weight.size(3);

    // Compute output dimensions
    int outH = (H - pad_h) * dilation_h + kH - pad_h * stride_h;
    int outW = (W - pad_w) * dilation_w + kW - pad_w * stride_w;

    // Allocate output tensor
    auto output = torch::empty({N, out_channels, outH, outW}, x.options());

    // Launch one block per (n, c) pair
    dim3 grid(N, out_channels);
    const int threads = 256;

    conv2d_kernel<<<grid, threads>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        N, in_channels, H, W,
        out_channels, kH, kW,
        stride_h, stride_w, pad_h, pad_w,
        dilation_h, dilation_w
    );

    return output;
}"""

cpp_source = """torch::Tensor forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    int64_t stride_h,
    int64_t stride_w,
    int64_t pad_h,
    int64_t pad_w,
    int64_t dilation_h,
    int64_t dilation_w);"""

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
