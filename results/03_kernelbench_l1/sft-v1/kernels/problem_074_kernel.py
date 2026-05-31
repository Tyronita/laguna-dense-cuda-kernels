import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = """#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// CUDA kernel for transposed 1D convolution with stride, padding, and dilation
// This kernel uses grid-stride loops to cover the entire output tensor.
// It computes the output for each (n, c) slice, where each slice is a (n, c) pair of indices
// and the corresponding output element is computed from the convolution.

template <typename scalar_t>
__global__ void conv_transpose_kernel_stride(
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    const int64_t stride,
    const int64_t padding,
    const int64_t dilation,
    const int64_t in_channels,
    const int64_t in_length,
    const int64_t out_channels,
    const int64_t kernel_size) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = in_length * out_channels;
    if (idx >= total_elements) return;

    // Decompose the index into a 2D coordinate
    int c = idx % out_channels;
    int n = (idx / out_channels);

    // Compute the output position for this (n, c) slice
    int out_idx = n * out_channels + c;
    int out_length = (in_length - 1) * stride + kernel_size - 2 * padding + dilation * kernel_size;

    scalar_t sum = 0;
    for (int k = 0; k < kernel_size; ++k) {
        int in_idx = (n * in_length + (k + padding)) * stride + (k * dilation);
        if (in_idx < in_length) continue;
        int in_channel = in_idx / in_length;
        int weight_idx = c * in_channels + in_channel;
        sum += input[in_idx] * weight[weight_idx];
    }

    if (bias != 0) {
        sum += bias[c];
    }

    output[out_idx] = sum;
}

// PyTorch wrapper function
torch::Tensor forward_cuda(
    torch::Tensor input,
    int64_t stride,
    int64_t padding,
    int64_t dilation,
    torch::Tensor weight,
    torch::Tensor bias) {

    auto out_length = (input.size(3) - 1) * stride +
                      (input.size(2) - 1) * padding +
                      dilation * input.size(2);

    auto output = torch::empty({input.size(0), input.size(1), out_length}, input.options());

    const int threads = 256;
    const int blocks = (input.numel() * input.size(1) + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "conv_transpose_kernel_stride", ([&] {
        conv_transpose_kernel_stride<scalar_t><<<blocks, threads>>>(
            output.data_ptr<scalar_t>(),
            input.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            bias.data_ptr<scalar_t>(),
            stride,
            padding,
            dilation,
            input.size(1),
            input.size(3),
            input.size(1),
            input.size(2)
        );
    }));

    return output;
}"""

cpp_source = """torch::Tensor forward_cuda(
    torch::Tensor input,
    int64_t stride,
    int64_t padding,
    int64_t dilation,
    torch::Tensor weight,
    torch::Tensor bias);"""

custom_ops = load_inline(
    name="custom_ops",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['forward_cuda'],
    verbose=False,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"],
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x):
        return custom_ops.forward_cuda(x)
