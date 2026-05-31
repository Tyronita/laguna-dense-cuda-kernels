import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = """#include <torch/extension.h>
#include <cuda_runtime.h>

// Custom CUDA kernel for 1D convolution with stride/dilation
__global__ void conv1d_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int batch_size,
    int in_channels,
    int in_length,
    int out_channels,
    int kernel_size,
    int stride,
    int dilation) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * out_channels * in_length;
    if (idx < total) {
        int b = idx / (in_channels * out_channels);
        int c = (idx / in_length) % out_channels;
        int l = idx % in_length;

        float sum = bias[c];
        for (int ic = 0; ic < in_channels; ic++) {
            for (int kh = 0; kh < kernel_size; kh++) {
                int in_idx = b * (in_channels * in_length) + ic * in_length + l * stride + kh * dilation;
                if (in_idx >= 0 && in_idx < in_length) {
                    sum += input[in_idx] * weight[c * in_channels + ic * kernel_size + kh];
                }
            }
        }
        output[idx] = sum;
    }
}

torch::Tensor conv1d_cuda(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, int stride, int dilation) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "Weight must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "Bias must be a CUDA tensor");

    int batch_size = x.size(0);
    int in_channels = x.size(1);
    int in_length = x.size(2);
    int out_channels = weight.size(0);
    int kernel_size = weight.size(2);

    int out_length = (in_length - 1) * stride + kernel_size * dilation;
    TORCH_CHECK(out_length > 0, "Output length must be positive");

    auto output = torch::empty({batch_size, out_channels, out_length}, x.options());

    const int block_size = 256;
    const int num_blocks = (batch_size * out_channels * out_length + block_size - 1) / block_size;

    conv1d_kernel<<<num_blocks, block_size>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        in_length,
        out_channels,
        kernel_size,
        stride,
        dilation
    );

    return output;
}

torch::Tensor forward(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, int stride, int dilation) {
    return conv1d_cuda(x, weight, bias, stride, dilation);
}"""

cpp_source = """torch::Tensor conv1d_cuda(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, int stride, int dilation);\ntorch::Tensor forward(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, int stride, int dilation);"""

custom_ops = load_inline(
    name="custom_ops",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['conv1d_cuda', 'forward'],
    verbose=False,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"],
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x):
        return custom_ops.conv1d_cuda(x)
