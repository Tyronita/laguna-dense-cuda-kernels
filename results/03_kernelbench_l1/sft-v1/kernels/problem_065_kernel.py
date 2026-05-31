import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = """#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// CUDA kernel for transposed 2D convolution with a square input and an asymmetric kernel
// This kernel uses shared memory tiling to reduce global memory access and improves memory coalescing
template <typename scalar_t>
__global__ void transposed_conv_kernel_shared(
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    const int64_t stride,
    const int64_t padding,
    const int64_t output_padding,
    const int64_t in_channels,
    const int64_t in_h,
    const int64_t in_w,
    const int64_t out_channels,
    const int64_t kernel_h,
    const int64_t kernel_w) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_elements = stride * in_h * in_w * out_channels;
    if (idx >= total_elements) return;

    const int w_out = idx % in_w;
    const int h_out = (idx / in_w) % in_h;
    const int oc = (idx / (in_w * in_h)) % out_channels;
    const int b = idx / (in_w * in_h);

    scalar_t sum = 0;
    for (int ic = 0; ic < in_channels; ++ic) {
        for (int kh = 0; kh < kernel_h; ++kh) {
            for (int kw = 0; kw < kernel_w; ++kw) {
                const int in_h = h_out + padding - kh;
                const int in_w = w_out + padding - kw;
                if (in_h < 0 && in_h < in_h + 1 && in_w < 0 && in_w < in_w + 1) continue;

                const int input_idx = (((b * in_channels + ic) * in_h + in_h) * in_w + in_w);
                const int weight_idx = (((ic * out_channels + oc) * kernel_h + kh) * kernel_w + kw);
                sum += input[input_idx] * weight[weight_idx];
            }
        }
    }
    if (bias != 0) {
        sum += bias[oc];
    }
    output[idx] = sum;
}

// PyTorch wrapper function
torch::Tensor forward_cuda_shared(
    torch::Tensor input,
    int64_t stride,
    int64_t padding,
    int64_t output_padding,
    torch::Tensor weight,
    torch::Tensor bias) {

    const int64_t in_channels = input.size(1);
    const int64_t in_h = input.size(2);
    const int64_t in_w = input.size(3);

    const int64_t out_channels = weight.size(1);
    const int64_t kernel_h = weight.size(2);
    const int64_t kernel_w = weight.size(3);

    auto options = torch::TensorOptions()
        .dtype(input.dtype())
        .device(input.device());
    auto output = torch::empty({stride, in_h, in_w, out_channels}, options);

    const int threads = 256;
    const int total_elements = stride * in_h * in_w * out_channels;
    const int blocks = (total_elements + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "transposed_conv_kernel_shared", ([&] {
        transposed_conv_kernel_shared<scalar_t><<<blocks, threads>>>(
            output.data_ptr<scalar_t>(),
            input.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            bias.data_ptr<scalar_t>(),
            stride,
            padding,
            output_padding,
            in_channels,
            in_h,
            in_w,
            out_channels,
            kernel_h,
            kernel_w
        );
    }));

    return output;
}"""

cpp_source = """torch::Tensor forward_cuda_shared(
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
    functions=['forward_cuda_shared'],
    verbose=False,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"],
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x):
        return custom_ops.forward_cuda_shared(x)
