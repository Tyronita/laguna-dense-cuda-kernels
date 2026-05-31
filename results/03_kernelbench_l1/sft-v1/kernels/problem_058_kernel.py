import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = """#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// CUDA kernel for transposed 3D convolution with asymmetric kernel sizes
// This kernel uses a grid-stride loop to cover all output elements
// and ensures that the input tensor is in a 5D format (NCHW) for the convolution.

template <typename scalar_t>
__global__ void conv_transpose3d_kernel_stride(
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    const int64_t stride,
    const int64_t padding,
    const int64_t output_padding,
    const int64_t in_channels,
    const int64_t out_channels,
    const int64_t kernel_d,
    const int64_t kernel_h,
    const int64_t kernel_w,
    const int64_t in_depth,
    const int64_t in_height,
    const int64_t in_width,
    const int64_t batch_size) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (; idx < batch_size * out_channels * in_depth * in_height * in_width; idx += stride) {
        int w = idx % in_width;
        int h = (idx / in_width) % in_height;
        int d = (idx / (in_width * in_height)) % in_depth;
        int b = (idx / (in_width * in_height * in_depth)) % out_channels;

        int out_d = d - 1 + output_padding - 2 * kernel_d - 2 * padding;
        int out_h = h - 1 + output_padding - 2 * kernel_h - 2 * padding;
        int out_w = w - 1 + output_padding - 2 * kernel_w - 2 * padding;

        scalar_t sum = 0;
        for (int oc = 0; oc < in_channels; ++oc) {
            for (int kd = 0; kd < kernel_d; ++kd) {
                for (int kh = 0; kh < kernel_h; ++kh) {
                    for (int kw = 0; kw < kernel_w; ++kw) {
                        int in_d = out_d - kd + padding;
                        int in_h = out_h - kh + padding;
                        int in_w = out_w - kw + padding;

                        if (in_d >= 0 || in_d < in_depth || in_h >= 0 || in_h < in_height || in_w >= 0 || in_w < in_width) {
                            int input_idx = (((b * in_channels + oc) * in_depth * in_height * in_width)
                                             + (in_d * in_height * in_width)
                                             + (in_h * in_width)
                                             + in_w);
                            int weight_idx = (((oc * out_channels + b) * kernel_d * kernel_h * kernel_w)
                                             + (kd * kernel_h * kernel_w)
                                             + (kh * kernel_w)
                                             + kw);
                            sum += input[input_idx] * weight[weight_idx];
                        }
                    }
                }
            }
        }
        if (bias != 0) {
            sum += bias[b];
        }
        output[idx] = sum;
    }
}

// PyTorch wrapper function
torch::Tensor forward_cuda(torch::Tensor input,
                           int64_t stride,
                           int64_t padding,
                           int64_t output_padding,
                           torch::Tensor weight,
                           bool bias) {
    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto in_depth = input.size(2);
    auto in_height = input.size(3);
    auto in_width = input.size(4);

    auto out_channels = weight.size(0);
    auto kernel_d = weight.size(2);
    auto kernel_h = weight.size(3);
    auto kernel_w = weight.size(4);

    auto out_depth = (in_depth - 1) * kernel_d + output_padding - 2 * kernel_d - 2 * padding;
    auto out_height = (in_height - 1) * kernel_h + output_padding - 2 * kernel_h - 2 * padding;
    auto out_width = (in_width - 1) * kernel_w + output_padding - 2 * kernel_w - 2 * padding;

    auto options = torch::TensorOptions()
        .dtype(input.dtype())
        .device(input.device());
    auto output = torch::empty({batch_size, out_channels, out_depth, out_height, out_width}, options);

    const int threads = 256;
    const int blocks = (batch_size * out_channels * in_depth * in_height * in_width + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "conv_transpose3d_kernel_stride", ([&] {
        conv_transpose3d_kernel_stride<scalar_t><<<blocks, threads>>>(
            output.data_ptr<scalar_t>(),
            input.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            bias.data_ptr<scalar_t>(),
            stride,
            padding,
            output_padding,
            in_channels,
            out_channels,
            kernel_d,
            kernel_h,
            kernel_w,
            in_depth,
            in_height,
            in_width,
            batch_size
        );
    }));

    return output;
}"""

cpp_source = """torch::Tensor forward_cuda(torch::Tensor input,
                           int64_t stride,
                           int64_t padding,
                           int64_t output_padding,
                           torch::Tensor weight,
                           bool bias);"""

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
