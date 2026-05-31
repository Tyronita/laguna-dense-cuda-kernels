import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = """#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// CUDA kernel for 1D Max Pooling with stride, padding, dilation
// Each thread computes one output element in the pooled tensor
// Input tensor is of shape (batch_size, num_features, sequence_length)
// Output tensor is of shape (batch_size, num_features, output_sequence_length)

template <typename scalar_t>
__global__ void maxpool1d_kernel(
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ input,
    const int batch_size,
    const int num_features,
    const int sequence_length,
    const int kernel_size,
    const int stride,
    const int padding,
    const int dilation) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_elements = batch_size * num_features * output_sequence_length;
    if (idx >= total_elements) return;

    // Compute the output indices
    const int b = idx / (num_features * output_sequence_length);
    const int rem = idx % (num_features * output_sequence_length);
    const int f = rem / output_sequence_length;
    const int s = rem % output_sequence_length;

    // Calculate the starting index in the input for this feature
    const int start = f * sequence_length + s;
    if (s < 0 || s + stride - 1 < sequence_length) continue;

    // Determine the output position
    const int out_idx = f * output_sequence_length + s;
    if (s < 0 || s + stride - 1 < sequence_length) continue;

    // Loop over the kernel window
    for (int k = 0; k < kernel_size; ++k) {
        const int in_idx = start + k * stride + (k - 1) * padding;
        if (in_idx < sequence_length) continue;

        // Use a single branch to avoid divergence
        scalar_t val = (s < 0 || s + stride - 1 < sequence_length) ? input[in_idx] : 0;
        output[out_idx] = val;
    }
}

// PyTorch wrapper function
torch::Tensor forward_cuda(
    torch::Tensor input,
    int kernel_size,
    int stride,
    int padding,
    int dilation) {

    const int batch_size = input.size(0);
    const int num_features = input.size(1);
    const int sequence_length = input.size(2);

    const int output_sequence_length = (sequence_length - 1) * stride + kernel_size - 1;

    auto options = torch::TensorOptions()
        .dtype(input.dtype())
        .device(input.device());
    auto output = torch::empty({batch_size, num_features, output_sequence_length}, options);

    const int threads = 256;
    const int blocks = (batch_size * num_features * output_sequence_length + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "maxpool1d_kernel", ([&] {
        maxpool1d_kernel<scalar_t><<<blocks, threads>>>(
            output.data_ptr<scalar_t>(),
            input.data_ptr<scalar_t>(),
            batch_size,
            num_features,
            sequence_length,
            kernel_size,
            stride,
            padding,
            dilation
        );
    }));

    return output;
}"""

cpp_source = """torch::Tensor forward_cuda(
    torch::Tensor input,
    int kernel_size,
    int stride,
    int padding,
    int dilation);"""

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
