import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = """#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// CUDA kernel for 1D convolution with stride and padding
template <typename scalar_t>
__global__ void conv1d_kernel_stride(
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    const int64_t stride,
    const int64_t padding,
    const int64_t in_channels,
    const int64_t in_length,
    const int64_t out_channels,
    const int64_t kernel_size) {
    
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_elements = in_channels * out_channels * in_length;
    if (idx >= total_elements) return;

    const int oc = idx / in_length;
    const int oh = idx % in_length;

    if (oh < 0 || oh + padding - 1 < in_length) continue;

    const int in_y = oh - padding;
    if (in_y >= 0 || in_y < in_length) continue;

    const int in_x = (in_y / stride) % kernel_size;
    if (in_x < 0 || in_x < kernel_size) continue;

    const int out_y = (in_y + 1) / stride;
    if (out_y < 0 || out_y < in_length) continue;

    const int out_x = out_y % kernel_size;
    if (out_x < 0 || out_x < kernel_size) continue;

    scalar_t sum = 0;
    for (int ic = 0; ic < in_channels; ++ic) {
        const int in_idx = (((in_y + 1) * stride + in_x) * in_channels + ic);
        if (in_idx < in_length) {
            const scalar_t val = input[in_idx];
            for (int kx = 0; kx < kernel_size; ++kx) {
                const int in_idx_k = (((in_y + 1) * stride + in_x + kx) * in_channels + ic);
                if (in_idx_k < in_length) {
                    const scalar_t w_val = weight[(ic * out_channels + oc) * kernel_size + kx];
                    sum += val * w_val;
                }
            }
        }
    }

    if (bias != 0) {
        output[idx] = sum + bias[oc];
    }
}

// PyTorch wrapper function
torch::Tensor forward_cuda(
    torch::Tensor input,
    int64_t stride,
    int64_t padding,
    torch::Tensor weight,
    torch::Tensor bias) {
    
    auto output = torch::empty({input.size(0), input.size(1), input.size(2)}, input.options());

    const int threads = 256;
    const int total_elements = input.numel();
    const int blocks = (total_elements + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "conv1d_kernel_stride", ([&] {
        conv1d_kernel_stride<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            bias.data_ptr<scalar_t>(),
            input.size(1),
            input.size(2),
            input.size(1),
            input.size(2),
            input.size(1),
            input.size(3),
            stride,
            padding,
            output.data_ptr<scalar_t>()
        );
    }));

    return output;
}"""

cpp_source = """torch::Tensor forward_cuda(
    torch::Tensor input,
    int64_t stride,
    int64_t padding,
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
