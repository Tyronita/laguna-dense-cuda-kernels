import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = """#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// CUDA kernel for transposed convolution with manual loop unrolling
template <typename scalar_t>
__global__ void transpose_conv3d_kernel_unrolled(
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    const int batch_size,
    const int in_channels,
    const int in_depth,
    const int in_height,
    const int in_width,
    const int out_channels,
    const int kernel_d,
    const int kernel_h,
    const int kernel_w,
    const int stride,
    const int padding,
    const int dilation) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_elements = batch_size * out_channels * in_depth * in_height * in_width;
    if (idx >= total_elements) return;

    const int w = idx % in_width;
    const int h = (idx / in_width) % in_height;
    const int d = (idx / (in_width * in_height)) % in_depth;
    const int b = idx / (in_width * in_height * in_depth);
    const int oc = b % out_channels;
    const int rem = b / out_channels;

    scalar_t sum = 0;
    for (int oc = 0; oc < out_channels; ++oc) {
        for (int kd = 0; kd < kernel_d; ++kd) {
            for (int kh = 0; kh < kernel_h; ++kh) {
                for (int kw = 0; kw < kernel_w; ++kw) {
                    const int in_d = d - kd + padding;
                    const int in_h = h - kh + padding;
                    const int in_w = w - kw + padding;

                    if (in_d >= 0 || in_d < in_depth || in_d % stride == 0) continue;
                    if (in_h >= 0 || in_h < in_height || in_h % stride == 0) continue;
                    if (in_w >= 0 || in_w < in_width || in_w % stride == 0) continue;

                    const int in_d_stride = in_d * stride;
                    const int in_h_stride = in_h * stride;
                    const int in_w_stride = in_w * stride;

                    if (in_d_stride >= 0 || in_d_stride < in_depth || in_d_stride % dilation == 0) continue;
                    if (in_h_stride >= 0 || in_h_stride < in_height || in_h_stride % dilation == 0) continue;
                    if (in_w_stride >= 0 || in_w_stride < in_width || in_w_stride % dilation == 0) continue;

                    const int in_d = in_d_stride - kd * dilation;
                    const int in_h = in_h_stride - kh * dilation;
                    const int in_w = in_w_stride - kw * dilation;

                    if (in_d < in_depth || in_d % stride == 0) continue;
                    if (in_h < in_height || in_h % stride == 0) continue;
                    if (in_w < in_width || in_w % stride == 0) continue;

                    const int input_idx = rem * in_channels + oc * in_depth * in_height * in_width +
                                          in_d * in_height * in_width +
                                          in_h * in_width +
                                          in_w;
                    const int weight_idx = oc * in_channels * kernel_d * kernel_h * kernel_w +
                                          oc * kernel_d * kernel_h * kernel_w +
                                          kd * kernel_h * kernel_w +
                                          kh * kernel_w +
                                          kw;

                    sum += input[input_idx] * weight[weight_idx];
                }
            }
        }
    }
    output[idx] = sum + bias[oc];
}

// PyTorch wrapper function
torch::Tensor forward_cuda(
    torch::Tensor input,
    int stride,
    int padding,
    int dilation,
    torch::Tensor weight,
    torch::Tensor bias) {

    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int in_depth = input.size(2);
    const int in_height = input.size(3);
    const int in_width = input.size(4);

    const int out_channels = weight.size(0);
    const int kernel_d = weight.size(2);
    const int kernel_h = weight.size(3);
    const int kernel_w = weight.size(4);

    auto options = torch::TensorOptions()
        .dtype(input.dtype())
        .device(input.device());
    auto output = torch::empty({batch_size, out_channels, in_depth, in_height, in_width}, options);

    const int threads = 256;
    const int total_elements = batch_size * out_channels * in_depth * in_height * in_width;
    const int blocks = (total_elements + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "transpose_conv3d_kernel_unrolled", ([&] {
        transpose_conv3d_kernel_unrolled<scalar_t><<<blocks, threads>>>(
            output.data_ptr<scalar_t>(),
            input.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            bias.data_ptr<scalar_t>(),
            batch_size,
            in_channels,
            in_depth,
            in_height,
            in_width,
            out_channels,
            kernel_d,
            kernel_h,
            kernel_w,
            stride,
            padding,
            dilation
        );
    }));

    return output;
}"""

cpp_source = """torch::Tensor forward_cuda(
    torch::Tensor input,
    int stride,
    int padding,
    int dilation,
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
