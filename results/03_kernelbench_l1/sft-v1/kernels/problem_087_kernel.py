import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = """#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// CUDA kernel for 2D convolution with bias addition
template <typename scalar_t>
__global__ void conv2d_bias_kernel(
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    const int64_t N,
    const int64_t in_channels,
    const int64_t in_h,
    const int64_t in_w,
    const int64_t out_channels) {
    
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_elements = N * out_channels * in_h * in_w;
    
    if (idx < total_elements) {
        const int w = idx % in_w;
        const int h = (idx / in_w) % in_h;
        const int oc = (idx / (in_w * in_h)) % out_channels;
        const int n = idx / (in_w * in_h);
        
        scalar_t sum = 0;
        for (int ic = 0; ic < in_channels; ++ic) {
            const int input_idx = (((n * in_channels + ic) * in_h + h) * in_w + w);
            const int weight_idx = (((ic * out_channels + oc) * in_h + h) * in_w + w);
            sum += input[input_idx] * weight[weight_idx];
        }
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
    
    auto output = torch::empty({input.size(0), input.size(1), input.size(2), input.size(3)}, input.options());
    
    const int threads = 256;
    const int total_elements = input.numel();
    const int blocks = (total_elements + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "conv2d_bias_kernel", ([&] {
        conv2d_bias_kernel<scalar_t><<<blocks, threads>>>(
            output.data_ptr<scalar_t>(),
            input.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            bias.data_ptr<scalar_t>(),
            input.size(0),
            input.size(1),
            input.size(2),
            input.size(3),
            input.size(1)
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
