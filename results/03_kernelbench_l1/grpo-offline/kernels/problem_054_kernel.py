import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = """#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Custom CUDA kernel for 3D convolution with optimized thread and block indexing
__global__ void conv3d_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int batch_size,
    int in_channels,
    int depth,
    int width,
    int height,
    int out_channels,
    int kernel_size,
    int stride,
    int padding) {

    // Calculate output dimensions
    int out_depth = (depth - 1) * stride - 2 * padding + kernel_size;
    int out_width = (width - 1) * stride - 2 * padding + kernel_size;
    int out_height = (height - 1) * stride - 2 * padding + kernel_size;

    // Calculate total output elements
    int total_output = batch_size * out_channels * out_depth * out_width * out_height;

    // Use 2D grid to better map threads to spatial dimensions
    int block_x = blockIdx.x;  // batch index
    int block_y = blockIdx.y;  // output channel index
    int block_z = blockIdx.z;  // output spatial index

    // Calculate thread index within the block
    int thread_id = threadIdx.x;
    int thread_stride = blockDim.x;

    // Process multiple elements per thread
    for (int idx = thread_id; idx < total_output; idx += thread_stride) {
        // Decode output indices
        int b = block_x;
        int c = block_y;
        int d = block_z / (out_width * out_height);
        int w = block_z % (out_width * out_height);
        int h = w / out_width;
        w = w % out_width;

        float sum = 0.0f;

        // Convolution computation
        for (int ic = 0; ic < in_channels; ic++) {
            for (int kh = 0; kh < kernel_size; kh++) {
                for (int kw = 0; kw < kernel_size; kw++) {
                    int in_d = d * stride - padding + kh;
                    int in_w = w * stride - padding + kw;
                    int in_h = h * stride - padding + kh;

                    if (in_d >= 0 && in_d < depth && in_w >= 0 && in_w < width && in_h >= 0 && in_h < height) {
                        int input_idx = b * (in_channels * depth * width * height) + 
                                       ic * (depth * width * height) + 
                                       in_d * (width * height) + 
                                       in_w * height + 
                                       in_h;
                        int weight_idx = c * (in_channels * kernel_size * kernel_size) + 
                                         ic * (kernel_size * kernel_size) + 
                                         kh * kernel_size + 
                                         kw;
                        sum += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }

        // Add bias
        sum += bias[c];

        // Write output
        int output_idx = b * (out_channels * out_depth * out_width * out_height) + 
                         c * (out_depth * out_width * out_height) + 
                         d * (out_width * out_height) + 
                         w * out_height + 
                         h;
        output[output_idx] = sum;
    }
}

// C++ interface exposed to PyTorch
torch::Tensor forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    int64_t stride,
    int64_t padding) {

    // Input dimensions
    int batch_size = x.size(0);
    int in_channels = x.size(1);
    int depth = x.size(2);
    int width = x.size(3);
    int height = x.size(4);

    // Output dimensions
    int out_channels = weight.size(0);
    int out_depth = (depth - 1) * stride - 2 * padding + weight.size(2);
    int out_width = (width - 1) * stride - 2 * padding + weight.size(3);
    int out_height = (height - 1) * stride - 2 * padding + weight.size(4);

    // Allocate output tensor
    auto output = torch::empty({batch_size, out_channels, out_depth, out_width, out_height}, x.options());

    // Configure 2D grid and 1D block
    dim3 blocks(
        batch_size,
        out_channels,
        out_depth * out_width * out_height
    );
    dim3 threads(256);

    // Launch kernel
    conv3d_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        depth,
        width,
        height,
        out_channels,
        weight.size(2),
        weight.size(3),
        padding
    );

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error in conv3d_kernel: %s\n", cudaGetErrorString(err));
    }

    return output;
}"""

cpp_source = """torch::Tensor forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    int64_t stride,
    int64_t padding);"""

custom_ops = load_inline(
    name="custom_ops",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['forward'],
    verbose=False,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"],
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x):
        return custom_ops.forward(x)
