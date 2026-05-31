import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = """#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// CUDA kernel for 3D Max Pooling with stride, padding, dilation
// Each thread computes one output element in the 3D pooling result
// Input dimensions: (batch_size, channels, dim1, dim2, dim3)
// Output dimensions: (batch_size, channels, out_dim1, out_dim2, out_dim3)

template <typename scalar_t>
__global__ void maxpool3d_kernel(
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ input,
    const int64_t stride,
    const int64_t padding,
    const int64_t dilation,
    const int64_t kernel_size,
    const int64_t batch_size,
    const int64_t channels,
    const int64_t dim1,
    const int64_t dim2,
    const int64_t dim3) {

    // Compute the output position for this thread
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * channels * dim1 * dim2 * dim3;
    if (idx >= total_elements) return;

    // Convert the index to 3D coordinates
    int c = idx / (dim1 * dim2 * dim3);
    int rem = idx % (dim1 * dim2 * dim3);
    int d1 = rem / (dim2 * dim3);
    int d2 = rem % (dim2 * dim3);
    int d3 = d1 / dim2;

    // Calculate output dimensions
    int out_d1 = (dim1 - 1) * stride + kernel_size - 2 * padding;
    int out_d2 = (dim2 - 1) * stride + kernel_size - 2 * padding;
    int out_d3 = (dim3 - 1) * stride + kernel_size - 2 * padding;

    int out_idx = c * (out_d1 * out_d2 * out_d3) +
                  out_d1 * out_d2 * out_d3 +
                  d1 * out_d2 * out_d3 +
                  d2 * out_d3 +
                  d3;

    // Initialize the maximum value
    scalar_t max_val = 0;

    // Loop over the input window
    for (int i = 0; i < dim1; i++) {
        for (int j = 0; j < dim2; j++) {
            for (int k = 0; k < dim3; k++) {
                int in_d1 = i - padding;
                int in_d2 = j - padding;
                int in_d3 = k - padding;

                int in_idx = c * (dim1 * dim2 * dim3) +
                             in_d1 * (dim2 * dim3) +
                             in_d2 * dim3 +
                             in_d3;

                int in_d1_stride = in_d1 * stride;
                int in_d2_stride = in_d2 * stride;
                int in_d3_stride = in_d3 * stride;

                int out_d1 = out_d1 - in_d1_stride + kernel_size - 2 * padding;
                int out_d2 = out_d2 - in_d2_stride + kernel_size - 2 * padding;
                int out_d3 = out_d3 - in_d3_stride + kernel_size - 2 * padding;

                int out_idx = out_d1 * out_d2 * out_d3 +
                             d1 * out_d2 * out_d3 +
                             d2 * out_d3 +
                             d3;

                if (out_d1 < 0 || out_d2 < 0 || out_d3 < 0) {
                    int in_idx = in_idx + (in_d1_stride - 1) * kernel_size +
                               (in_d2_stride - 1) * kernel_size +
                               (in_d3_stride - 1) * kernel_size;
                    if (in_idx < in_idx + kernel_size - 1) {
                        max_val = input[in_idx];
                    } else {
                        max_val = 0;
                    }
                }
            }
        }
    }
    output[out_idx] = max_val;
}

// PyTorch wrapper function
torch::Tensor forward_cuda(torch::Tensor input, int64_t stride, int64_t padding, int64_t dilation, int64_t kernel_size) {
    auto batch_size = input.size(0);
    auto channels = input.size(1);
    auto dim1 = input.size(2);
    auto dim2 = input.size(3);
    auto dim3 = input.size(4);

    auto out_dim1 = (dim1 - 1) * stride + kernel_size - 2 * padding;
    auto out_dim2 = (dim2 - 1) * stride + kernel_size - 2 * padding;
    auto out_dim3 = (dim3 - 1) * stride + kernel_size - 2 * padding;

    auto options = torch::TensorOptions()
        .dtype(input.dtype())
        .device(input.device())
        .requires_grad(false);
    auto output = torch::empty({batch_size, channels, out_dim1, out_dim2, out_dim3}, options);

    const int threads = 256;
    const int total_elements = batch_size * channels * out_dim1 * out_dim2 * out_dim3;
    const int blocks = (total_elements + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "maxpool3d_kernel", ([&] {
        maxpool3d_kernel<scalar_t><<<blocks, threads>>>(
            output.data_ptr<scalar_t>(),
            input.data_ptr<scalar_t>(),
            stride,
            padding,
            dilation,
            kernel_size,
            batch_size,
            channels,
            dim1,
            dim2,
            dim3
        );
    }));

    return output;
}"""

cpp_source = """torch::Tensor forward_cuda(torch::Tensor input, int64_t stride, int64_t padding, int64_t dilation, int64_t kernel_size);"""

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
