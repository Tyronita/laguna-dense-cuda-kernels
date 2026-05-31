import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = """#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// CUDA kernel for 3D convolution with manual loop unrolling
template <typename scalar_t>
__global__ void conv3d_unroll_kernel(
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    const int64_t stride,
    const int64_t padding,
    const int64_t dilation,
    const int64_t in_channels,
    const int64_t in_depth,
    const int64_t in_width,
    const int64_t in_height,
    const int64_t out_channels,
    const int64_t out_depth,
    const int64_t out_width,
    const int64_t out_height,
    const int64_t kernel_d,
    const int64_t kernel_w,
    const int64_t kernel_h) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_elements = out_depth * out_width * out_height * out_channels;
    if (idx >= total_elements) return;

    const int w = idx % out_width;
    const int h = (idx / out_width) % out_height;
    const int d = (idx / (out_width * out_height)) % out_depth;
    const int oc = idx / (out_width * out_height * out_depth);

    scalar_t sum = 0;
    for (int ic = 0; ic < in_channels; ++ic) {
        for (int kd = 0; kd < kernel_d; ++kd) {
            for (int kw = 0; kw < kernel_w; ++kw) {
                for (int kh = 0; kh < kernel_h; ++kh) {
                    const int in_d = d - kd + padding;
                    const int in_w = w - kw + padding;
                    const int in_h = h - kh + padding;

                    if (in_d >= 0 && in_d < in_depth && in_w >= 0 && in_w < in_width && in_h >= 0 && in_h < in_height) {
                        const int in_idx = (((d * in_width * in_height) + w) * in_width + in_w)
                                        * in_height + in_h;
                                        const int weight_idx = (((ic * kernel_d * kernel_w * kernel_h) + kd) * kernel_w + kw) * kernel_h + kh;
                                        sum += input[in_idx] * weight[weight_idx];
                    }
                }
            }
        }
    }
    if (bias != 0) {
        sum += bias[oc];
    }
    output[idx] = sum;
}

// PyTorch wrapper function
torch::Tensor forward_cuda(
    torch::Tensor input,
    int64_t stride,
    int64_t padding,
    int64_t dilation,
    torch::Tensor weight,
    torch::Tensor bias) {

    const int threads = 256;
    const int total_elements = input.numel() * input.size(2) * input.size(3) * input.size(4) * weight.numel();
    const int blocks = (total_elements + threads - 1) / threads;

    auto output = torch::empty({input.size(0), weight.size(0), input.size(2), input.size(3), input.size(4)}, input.options());

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "conv3d_unroll_kernel", ([&] {
        conv3d_unroll_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            bias.data_ptr<scalar_t>(),
            input.size(0),
            input.size(1),
            input.size(2),
            input.size(3),
            input.size(4),
            weight.size(0),
            input.size(2),
            input.size(3),
            input.size(4),
            stride,
            padding,
            dilation,
            input.size(1),
            input.size(2),
            input.size(3),
            input.size(4),
            weight.size(2),
            weight.size(3),
            weight.size(4)
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
