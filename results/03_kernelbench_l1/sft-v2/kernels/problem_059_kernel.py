import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = """#include <torch/extension.h>
#include <cuda_runtime.h>

// Custom CUDA kernel for 3D convolution with stride=1, padding=0, dilation=1, groups=1.
// Assumes input tensor shape: [batch_size, in_channels, height, width, depth]
// and output tensor shape: [batch_size, out_channels, height_out, width_out, depth_out].

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

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * out_channels * height * width * depth;
    if (idx < total) {
        // Decode output indices
        int d_out = idx % depth;
        int temp = idx / depth;
        int h_out = temp / (width * out_channels);
        temp %= (width * out_channels);
        int c_out = temp / width;
        int w_out = temp % width;

        int h_in = h_out - padding;
        int w_in = w_out - padding;

        float sum = bias[c_out];

        // Loop over input channels and kernel elements
        for (int c_in = 0; c_in < in_channels; c_in++) {
            for (int kh = 0; kh < kernel_h; kh++) {
                for (int kw = 0; kw < kernel_w; kw++) {
                    int h_in_idx = h_in + kh;
                    int w_in_idx = w_in + kw;
                    if (h_in_idx >= 0 && h_in_idx < height && w_in_idx >= 0 && w_in_idx < width) {
                        int input_idx = c_in * (height * width * depth) + 
                                         h_in_idx * (width * depth) + 
                                         w_in_idx * depth + d_in;
                        int weight_idx = c_in * (out_channels * kernel_h * kernel_w) + 
                                         c_out * (kernel_h * kernel_w) + 
                                         kh * kernel_w + kw;
                        sum += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }
        output[idx] = sum;
    }
}

// The forward function exposed via PyBind11.
torch::Tensor forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    int64_t stride,
    int64_t padding) {

    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "bias must be a CUDA tensor");

    int batch_size = x.size(0);
    int in_channels = x.size(1);
    int height = x.size(2);
    int width = x.size(3);
    int depth = x.size(4);

    int out_channels = weight.size(0);
    int kernel_h = weight.size(2);
    int kernel_w = weight.size(3);

    int h_out = (height - 1) * stride - 2 * padding + kernel_h;
    int w_out = (width - 1) * stride - 2 * padding + kernel_w;
    int d_out = depth;

    TORCH_CHECK(h_out > 0 && h_out < height, "Output height must be valid");
    TORCH_CHECK(w_out > 0 && w_out < width, "Output width must be valid");
    TORCH_CHECK(d_out == d_out, "Output depth must match input depth");

    auto output = torch::empty({batch_size, out_channels, h_out, w_out, d_out}, x.options());

    int total = batch_size * out_channels * h_out * w_out * d_out;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    conv3d_kernel<<<blocks, threads>>>(
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
        kernel_h,
        kernel_w,
        stride,
        padding
    );

    return output;
}"""

cpp_source = """torch::Tensor forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    int64_t stride,
    int64_t padding);"""

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
