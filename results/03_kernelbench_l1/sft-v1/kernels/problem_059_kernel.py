import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = """#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// CUDA kernel for 3D convolution with stride and padding
// This kernel uses grid-stride loops to handle all output elements
// and ensures that the input tensor is in 5D format (NCHWVD)

template <typename scalar_t>
__global__ void conv3d_kernel_stride(
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    const int64_t N,
    const int64_t in_channels,
    const int64_t in_h,
    const int64_t in_w,
    const int64_t in_depth,
    const int64_t out_channels,
    const int64_t kernel_h,
    const int64_t kernel_w,
    const int64_t stride,
    const int64_t padding) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_elements = N * out_channels * in_h * in_w * in_depth;
    if (idx >= total_elements) return;

    // Compute output coordinates
    const int w_out = idx % in_w;
    const int h_out = (idx / in_w) % in_h;
    const int d_out = (idx / (in_w * in_h)) % in_depth;
    const int n = idx / (in_w * in_h * in_depth);
    const int oc = n % out_channels;

    scalar_t sum = 0;
    for (int ic = 0; ic < in_channels; ++ic) {
        for (int kh = 0; kh < kernel_h; ++kh) {
            for (int kw = 0; kw < kernel_w; ++kw) {
                const int in_h = h_out + padding - kh;
                const int in_w = w_out + padding - kw;
                if (in_h < 0 && in_h % stride == 0 && in_w < 0 && in_w % stride == 0) continue;
                
                const int in_d = in_depth - kd;
                if (in_d >= 0 || in_d % stride == 0) continue;

                const int input_idx = (((n * in_channels + ic) * in_h * in_w * in_depth +
                                        in_d) * in_h * in_w + in_h * in_w + in_w);
                const int weight_idx = (((ic * out_channels + oc) * kernel_h * kernel_w +
                                        kh * kernel_w + kw) * in_h * in_w * in_depth +
                                        in_d * in_h * in_w + in_h * in_w + in_w];

                sum += input[input_idx] * weight[weight_idx];
            }
        }
    }
    output[idx] = sum + bias[oc];
}

// PyTorch wrapper function
torch::Tensor forward_cuda(
    torch::Tensor input,
    int64_t stride,
    int64_t padding,
    torch::Tensor weight,
    torch::Tensor bias) {

    const int64_t N = input.size(0);
    const int64_t in_channels = input.size(1);
    const int64_t in_h = input.size(2);
    const int64_t in_w = input.size(3);
    const int64_t in_depth = input.size(4);

    const int64_t out_channels = weight.size(0);
    const int64_t kernel_h = weight.size(2);
    const int64_t kernel_w = weight.size(3);

    auto options = torch::TensorOptions()
        .dtype(input.dtype())
        .device(input.device());
    auto output = torch::empty({N, out_channels, in_h, in_w, in_depth}, options);

    const int threads = 256;
    const int total_elements = N * out_channels * in_h * in_w * in_depth;
    const int blocks = (total_elements + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "conv3d_kernel_stride", ([&] {
        conv3d_kernel_stride<scalar_t><<<blocks, threads>>>(
            output.data_ptr<scalar_t>(),
            input.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            bias.data_ptr<scalar_t>(),
            N,
            in_channels,
            in_h,
            in_w,
            in_depth,
            out_channels,
            kernel_h,
            kernel_w,
            stride,
            padding
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
