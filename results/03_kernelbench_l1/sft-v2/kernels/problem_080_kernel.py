import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = """#include <torch/extension.h>
#include <cuda.h>
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
    int kernel_h,
    int kernel_w,
    int stride,
    int padding,
    int dilation) {

    // Calculate output dimensions
    int out_height = (height - 1) * stride - 2 * padding + kernel_h * dilation;
    int out_width = (width - 1) * stride - 2 * padding + kernel_w * dilation;

    // Calculate output channel index
    int oc = blockIdx.z % out_channels;
    int b = blockIdx.z / out_channels;

    // Calculate output position
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;

    if (out_y < out_height && out_x < out_width) {
        float sum = bias[oc];
        
        // Convolution computation
        for (int ic = 0; ic < in_channels; ic++) {
            for (int kh = 0; kh < kernel_h; kh++) {
                for (int kw = 0; kw < kernel_w; kw++) {
                    int in_y = out_y * stride + kh * dilation;
                    int in_x = out_x * stride + kw * dilation;
                    
                    if (in_y >= 0 && in_y < height && in_x >= 0 && in_x < width) {
                        int input_idx = b * (in_channels * height * width) + 
                                       ic * (height * width) + 
                                       in_y * width + in_x;
                        int weight_idx = oc * (in_channels * kernel_h * kernel_w) + 
                                         ic * (kernel_h * kernel_w) + 
                                         kh * kernel_w + kw;
                        
                        sum += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }
        output[b * (out_channels * out_height * out_width) + 
               oc * (out_height * out_width) + 
               out_y * out_width + out_x] = sum;
    }
}

// C++ interface exposed to PyTorch
torch::Tensor forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    int64_t stride,
    int64_t padding,
    int64_t dilation) {
    
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "Weight must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "Bias must be a CUDA tensor");
    TORCH_CHECK(x.dim() == 4, "Input must be 4D");
    TORCH_CHECK(weight.dim() == 4, "Weight must be 4D");
    TORCH_CHECK(bias.dim() == 1, "Bias must be 1D");

    int batch_size = x.size(0);
    int in_channels = x.size(1);
    int height = x.size(2);
    int width = x.size(3);
    int out_channels = weight.size(0);
    int kernel_h = weight.size(2);
    int kernel_w = weight.size(3);

    // Calculate output dimensions
    int out_height = (height - 1) * stride - 2 * padding + kernel_h * dilation;
    int out_width = (width - 1) * stride - 2 * padding + kernel_w * dilation;

    auto output = torch::empty({batch_size, out_channels, out_height, out_width}, x.options());

    // Define block and grid sizes
    dim3 block(16, 16);
    dim3 grid((out_width + block.x - 1) / block.x,
              (out_height + block.y - 1) / block.y,
              batch_size * out_channels);

    const float* input_ptr = x.data_ptr<float>();
    const float* weight_ptr = weight.data_ptr<float>();
    const float* bias_ptr = bias.data_ptr<float>();
    float* output_ptr = output.data_ptr<float>();

    conv2d_kernel<<<grid, block>>>(
        input_ptr,
        weight_ptr,
        bias_ptr,
        output_ptr,
        batch_size,
        in_channels,
        height,
        width,
        out_channels,
        kernel_h,
        kernel_w,
        stride,
        padding,
        dilation
    );

    return output;
}"""

cpp_source = """torch::Tensor forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    int64_t stride,
    int64_t padding,
    int64_t dilation);"""

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
