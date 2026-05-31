import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = """#include <torch/extension.h>
#include <cuda_runtime.h>

// Custom CUDA kernel for 2D convolution with optimized thread and block indexing
__global__ void conv2d_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int batch_size,
    int in_channels,
    int height,
    int width,
    int out_channels,
    int kernel_size,
    int stride,
    int padding) {

    // Calculate output dimensions
    int out_height = (height - 1) * stride - 2 * padding + kernel_size;
    int out_width = (width - 1) * stride - 2 * padding + kernel_size;

    // Calculate output channel index
    int out_channel = blockIdx.x % out_channels;
    // Batch index from channels
    int b = blockIdx.x / out_channels;

    // Calculate output position
    int out_h = blockIdx.y * blockDim.y + threadIdx.y;
    int out_w = blockIdx.z * blockDim.z + threadIdx.z;

    if (out_h >= 0 && out_w >= 0) {
        float sum = bias[out_channel];
        
        // Loop over input channels
        for (int c = 0; c < in_channels; c++) {
            // Loop over input height and width
            for (int h = 0; h < height; h++) {
                for (int w = 0; w < width; w++) {
                    int in_h = h - padding;
                    int in_w = w - padding;
                    
                    if (in_h >= 0 && in_h < height && in_w >= 0 && in_w < width) {
                        int input_idx = b * (in_channels * height * width) + 
                                       c * (height * width) + 
                                       in_h * width + in_w;
                                    
                        int weight_idx = out_channel * (in_channels * kernel_size * kernel_size) + 
                                       c * (kernel_size * kernel_size) + 
                                       in_h * kernel_size + in_w;
                                    
                        sum += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }
        
        // Write output
        int output_idx = b * (out_channels * out_height * out_width) + 
                         out_channel * (out_height * out_width) + 
                         out_h * out_width + out_w;
        output[output_idx] = sum;
    }
}

// C++ interface exposed to PyTorch
torch::Tensor conv2d_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, int64_t stride, int64_t padding) {
    // Input dimensions
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int height = input.size(2);
    int width = input.size(3);
    
    // Output dimensions
    int out_channels = weight.size(0);
    int out_height = (height - 1) * stride - 2 * padding + weight.size(1);
    int out_width = (width - 1) * stride - 2 * padding + weight.size(2);
    
    // Create output tensor
    auto output = torch::empty({batch_size, out_channels, out_height, out_width}, input.options());
    
    // Define block and grid sizes
    const int block_size = 16;
    dim3 block(block_size, block_size);
    dim3 grid((batch_size * out_channels + block_size - 1) / block_size, 
              (out_height + block_size - 1) / block_size, 
              (out_width + block_size - 1) / block_size);
    
    // Launch kernel
    conv2d_kernel<<<grid, block>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        height,
        width,
        out_channels,
        weight.size(1),
        weight.size(2),
        stride,
        padding
    );
    
    return output;
}"""

cpp_source = """torch::Tensor conv2d_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, int64_t stride, int64_t padding);"""

custom_ops = load_inline(
    name="custom_ops",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['conv2d_cuda'],
    verbose=False,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"],
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x):
        return custom_ops.conv2d_cuda(x)
