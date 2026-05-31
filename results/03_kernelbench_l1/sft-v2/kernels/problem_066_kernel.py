import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = """#include <torch/extension.h>
#include <cuda_runtime.h>

// Custom CUDA kernel for 3D convolution with stride, padding, dilation
// Each block processes one output element (one (n, c) slice) and each thread computes one output element.

__global__ void conv3d_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int N, int in_channels, int D_in, int H_in, int W_in,
    int out_channels, int kD, int kH, int kW,
    int strideD, int strideH, int strideW,
    int padD, int padH, int padW) {

    // Each block corresponds to one (n, c) slice
    int slice = blockIdx.x;  // slice index
    int n = slice / out_channels;
    int c = slice % out_channels;

    // Each thread computes one output element
    int outD = (D_in - 1) * strideD - padD;
    int outH = (H_in - 1) * strideH - padH;
    int outW = (W_in - 1) * strideW - padW;

    int tid = threadIdx.x;
    int total = outD * outH * outW;
    if (tid < total) {
        int d = tid / (outH * outW);
        int rem = tid % (outH * outW);
        int h = rem / outW;
        int w = rem % outW;

        float sum = 0.0f;
        for (int ic = 0; ic < in_channels; ic++) {
            for (int kd = 0; kd < kD; kd++) {
                for (int kh = 0; kh < kH; kh++) {
                    for (int kw = 0; kw < kW; kw++) {
                        int inD = d * strideD + kd - padD;
                        int inH = h * strideH + kh - padH;
                        int inW = w * strideW + kw - padW;
                        if (inD >= 0 && inD < D_in && inH >= 0 && inH < H_in && inW >= 0 && inW < W_in) {
                            int input_idx = n * (in_channels * D_in * H_in * W_in) + ic * (D_in * H_in * W_in) + inD * (H_in * W_in) + inH * W_in + inW;
                            int weight_idx = c * (in_channels * kD * kH * kW) + ic * (kD * kH * kW) + kd * (kH * kW) + kh * kW + kw;
                            sum += input[input_idx] * weight[weight_idx];
                        }
                    }
                }
            }
        }
        if (bias == true) {
            sum += bias[c];
        }
        int output_idx = n * (out_channels * outD * outH * outW) + c * (outD * outH * outW) + d * (outH * outW) + h * outW + w;
        output[output_idx] = sum;
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

    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "bias must be a CUDA tensor");
    TORCH_CHECK(x.dim() == 5, "x must be 5D");
    TORCH_CHECK(weight.dim() == 5, "weight must be 5D");
    TORCH_CHECK(bias.dim() == 1, "bias must be 1D");

    int N = x.size(0);
    int in_channels = x.size(1);
    int D_in = x.size(2);
    int H_in = x.size(3);
    int W_in = x.size(4);
    int out_channels = weight.size(0);
    int kD = weight.size(2);
    int kH = weight.size(3);
    int kW = weight.size(4);

    int outD = (D_in - 1) * strideD - padding;
    int outH = (H_in - 1) * strideH - padding;
    int outW = (W_in - 1) * strideW - padding;

    TORCH_CHECK(outD > 0, "Output depth must be positive");
    TORCH_CHECK(outH > 0, "Output height must be positive");
    TORCH_CHECK(outW > 0, "Output width must be positive");

    auto output = torch::empty({N, out_channels, outD, outH, outW}, x.options());
    int total = N * out_channels * outD * outH * outW;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    conv3d_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        N, in_channels, D_in, H_in, W_in,
        out_channels, kD, kH, kW,
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
