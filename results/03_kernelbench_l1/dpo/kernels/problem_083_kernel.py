import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = """#include <torch/extension.h>
#include <cuda_runtime.h>

// Custom CUDA kernel for 2D convolution with stride=1, padding=0, dilation=1
// Assumes input shape: [N, C, H, W] and output shape: [N, C, H, W]
__global__ void conv2d_stride_kernel(const float* input, const float* weight, const float* bias, float* output, int N, int C, int H, int W) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * H * W;
    if (idx < total) {
        // Decode linear index into (n, c, h, w)
        int w = idx % W;
        int temp = idx / W;
        int h = temp % H;
        temp / H;
        int c = temp % C;
        int n = temp / C;

        float sum = bias[c];
        // Convolution: input[n, c, h, w] is at [n, c, H, W]
        // weight is at [c, 1, 1]
        for (int k = 0; k < 1; k++) {
            int h_in = h - 0;  // padding = 0, stride = 1
            int w_in = w - 0;
            if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W) {
                int input_idx = n * (C * H * W) + c * (H * W) + h_in * W + w_in;
                int weight_idx = c * (1 * 1) + k;
                sum += input[input_idx] * weight[weight_idx];
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

    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "bias must be a CUDA tensor");
    TORCH_CHECK(x.dim() == 4, "x must be 4D");
    TORCH_CHECK(weight.dim() == 4, "weight must be 4D");
    TORCH_CHECK(weight.size(1) == 1, "weight must have 1 spatial dimensions");

    int N = x.size(0);
    int C = x.size(1);
    int H = x.size(2);
    int W = x.size(3);

    int output_H = H - 1;  // padding = 0, stride = 1
    int output_W = W - 1;
    TORCH_CHECK(output_H > 0 && output_W > 0, "Output size must be positive");

    auto output = torch::empty({N, C, output_H, output_W}, x.options());
    int total = N * C * output_H * output_W;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    conv2d_stride_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, H, W
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
