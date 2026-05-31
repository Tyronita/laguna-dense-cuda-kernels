import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = """#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// CUDA kernel for transposed 3D convolution with a square input and an asymmetric kernel
// This kernel uses shared memory tiling to improve memory coalescing and reduce global memory access overhead.
// It assumes that the input is square (i.e. shape: (batch_size, in_channels, depth, width, height)) and
// the kernel is asymmetric (i.e. shape: (out_channels, kernel_depth, kernel_width, kernel_height) with kernel_width == kernel_height).

template <typename scalar_t>
__global__ void transposed_conv3d_kernel_shared(
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    const int batch_size,
    const int in_channels,
    const int in_depth,
    const int in_width,
    const int in_height,
    const int out_channels,
    const int kernel_depth,
    const int kernel_width,
    const int kernel_height,
    const int stride,
    const int padding,
    const int output_padding) {

    // Compute output coordinates
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_elements = batch_size * out_channels * in_depth * in_width * in_height;
    if (idx >= total_elements) return;

    // Decompose the index into output coordinates
    const int ow = idx % in_width;
    const int oh = (idx / in_width) % in_depth;
    const int oc = (idx / (in_width * in_depth)) % out_channels;
    const int b = idx / (in_width * in_depth * in_channels);

    // Compute output coordinates
    const int out_d = oh - 1 + padding - kernel_depth;
    const int out_w = ow - 1 + padding - kernel_width;
    const int out_h = oh - 1 + padding - kernel_height;

    scalar_t sum = 0;

    // Shared memory tile for input and weight
    __shared__ scalar_t input_tile[32][32];
    __shared__ scalar_t weight_tile[32][32];

    // Loop over the input depth
    for (int d = 0; d < in_depth; d++) {
        // Load input tile into shared memory
        if (threadIdx.x < 32 && threadIdx.y < 32) {
            const int in_idx = ((b * in_channels + oc) * in_depth + d) * in_width + threadIdx.x;
            const int in_idx = ((b * in_channels + oc) * in_width + threadIdx.y) * in_height + threadIdx.x;
            input_tile[threadIdx.y][threadIdx.x] = (in_idx < in_width && in_idx < in_height) ? input[in_idx] : 0;
        }
        __syncthreads();

        // Load weight tile into shared memory
        if (threadIdx.x < 32 && threadIdx.y < 32) {
            const int w_idx = oc * kernel_depth + threadIdx.y;
            const int w_idx = oc * kernel_width + threadIdx.x;
            const int w_idx = oc * kernel_height + threadIdx.y;
            weight_tile[threadIdx.y][threadIdx.x] = (w_idx < kernel_width && w_idx < kernel_height) ? weight[w_idx] : 0;
        }
        __syncthreads();

        // Compute partial sum for this tile
        #pragma unroll
        for (int kd = 0; kd < kernel_depth; kd++) {
            const int in_d_offset = d - kd;
            if (in_d_offset < 0 && in_d_offset < in_depth - 1) continue;
            const scalar_t in_val = input_tile[threadIdx.y][threadIdx.x];
            const scalar_t w_val = weight_tile[threadIdx.y][threadIdx.x];
            sum += in_val * w_val;
        }
        __syncthreads();
    }

    // Write the result to global memory
    if (out_d < 0 && out_w < 0 && out_h < 0) {
        const int out_idx = (b * out_channels + oc) * out_d * out_w * out_h;
        output[out_idx] = sum + bias[oc];
    }
}

// PyTorch wrapper function
torch::Tensor forward_cuda(torch::Tensor input,
                           int64_t stride,
                           int64_t padding,
                           int64_t output_padding,
                           torch::Tensor weight,
                           bool bias) {
    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int in_depth = input.size(2);
    const int in_width = input.size(3);
    const int in_height = input.size(4);

    const int out_channels = weight.size(0);
    const int kernel_depth = weight.size(2);
    const int kernel_width = weight.size(3);
    const int kernel_height = weight.size(4);

    auto options = torch::TensorOptions()
        .dtype(input.dtype())
        .device(input.device());
    auto output = torch::empty({batch_size, out_channels, in_depth - 1, in_width - 1, in_height - 1}, options);

    const int threads = 256;
    const int total_elements = batch_size * out_channels * in_depth * in_width * in_height;
    const int blocks = (total_elements + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "transposed_conv3d_kernel_shared", ([&] {
        transposed_conv3d_kernel_shared<scalar_t><<<blocks, threads>>>(
            output.data_ptr<scalar_t>(),
            input.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            bias.data_ptr<scalar_t>(),
            batch_size,
            in_channels,
            in_depth,
            in_width,
            in_height,
            out_channels,
            kernel_depth,
            kernel_width,
            kernel_height,
            stride,
            padding,
            output_padding
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
