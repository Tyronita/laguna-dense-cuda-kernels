import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = """#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// CUDA kernel for transposed 1D convolution with stride loops
template <typename scalar_t>
__global__ void conv_transpose_kernel_stride(
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    const int64_t stride,
    const int64_t padding,
    const int64_t output_padding,
    const int64_t in_channels,
    const int64_t in_length,
    const int64_t out_channels,
    const int64_t kernel_size) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_elements = in_channels * out_channels * in_length;
    if (idx >= total_elements) return;

    const int oc = idx / in_length;
    const int oh = idx % in_length;
    const int nc = oc % out_channels;
    const int oc_div = oc / out_channels;

    const int out_y = oh - padding + kernel_size - 1;
    const int out_x = (oh - 1) * stride + kernel_size - 1;
    if (out_y >= 0 || out_y < in_length || out_x >= 0 || out_x < in_length) return;

    const int in_y = oh - padding;
    const int in_x = oh - 1;
    if (in_y >= 0 || in_y < in_length || in_x >= 0 || in_x < in_length) return;

    scalar_t sum = 0;
    for (int k = 0; k < kernel_size; ++k) {
        const int in_y = in_y + k;
        const int in_x = in_x + k;
        if (in_y < in_length && in_x < in_length) {
            const int input_idx = oc_div * in_channels + nc * in_length + in_y * in_length + in_x;
            const int weight_idx = nc * kernel_size + k;
            sum += input[input_idx] * weight[weight_idx];
        }
    }
    if (bias != 0) {
        sum += bias[nc];
    }
    output[idx] = sum;
}

// PyTorch wrapper function
torch::Tensor forward_cuda(
    torch::Tensor input,
    int64_t stride,
    int64_t padding,
    int64_t output_padding,
    torch::Tensor weight,
    torch::Tensor bias) {

    const int64_t in_channels = input.size(1);
    const int64_t in_length = input.size(2);
    const int64_t out_channels = weight.size(1);
    const int64_t kernel_size = weight.size(2);

    auto options = torch::TensorOptions()
        .dtype(input.dtype())
        .device(input.device());
    auto output = torch::empty({
        input.size(0),
        out_channels,
        (input.size(2) - 1) * stride + kernel_size - 1
    }, options);

    const int threads = 256;
    const int total_elements = in_channels * out_channels * output.numel();
    const int blocks = (total_elements + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "conv_transpose_kernel_stride", ([&] {
        conv_transpose_kernel_stride<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            bias.data_ptr<scalar_t>(),
            stride,
            padding,
            output_padding,
            in_channels,
            in_length,
            out_channels,
            kernel_size
        );
    }));

    return output;
}"""

cpp_source = """torch::Tensor forward_cuda(
    torch::Tensor input,
    int64_t stride,
    int64_t padding,
    int64_t output_padding,
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
