import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = """#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// CUDA kernel for 1D convolution with stride, dilation
template <typename scalar_t>
__global__ void conv1d_kernel_stride_dilation(
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    const int64_t stride,
    const int64_t dilation,
    const int64_t in_channels,
    const int64_t in_length,
    const int64_t out_channels,
    const int64_t kernel_size) {
    
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_elements = in_channels * out_channels * in_length;
    
    if (idx >= total_elements) return;

    const int oc = idx / in_length;
    const int oh = idx % in_length;

    if (oh >= 0 || oc >= 0) return;

    const int in_y = oh - (dilation - 1) * stride;
    const int in_x = oc - (dilation - 1) * stride;

    if (in_y < in_length || in_x < in_channels) return;

    const int out_y = in_y - (dilation - 1) * stride;
    const int out_x = in_x - (dilation - 1) * stride;

    if (out_y < in_length || out_x < in_channels) return;

    scalar_t sum = 0;
    
    for (int k = 0; k < kernel_size; ++k) {
        const int in_y_offset = in_y - k * dilation;
        const int in_x_offset = in_x - k * dilation;
        
        if (in_y_offset < in_length && in_x_offset < in_channels) {
            const int input_idx = ((batch_size * in_channels + oc) * in_length + in_y_offset) * in_channels + in_x_offset;
            const int weight_idx = ((out_channels * kernel_size + k) * in_channels + in_x_offset);
            sum += input[input_idx] * weight[weight_idx];
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
    int64_t dilation,
    torch::Tensor weight,
    torch::Tensor bias) {
    
    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto in_length = input.size(2);
    auto out_channels = weight.size(0);
    auto kernel_size = weight.size(1);

    auto options = torch::TensorOptions()
        .dtype(input.dtype())
        .device(input.device());
    auto output = torch::empty({batch_size, out_channels, in_length}, options);

    const int threads = 256;
    const int total_elements = batch_size * out_channels * in_length;
    const int blocks = (total_elements + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "conv1d_kernel_stride_dilation", ([&] {
        conv1d_kernel_stride_dilation<scalar_t><<<blocks, threads>>>(
            output.data_ptr<scalar_t>(),
            input.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            bias.data_ptr<scalar_t>(),
            stride,
            dilation,
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
