import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = """#include <torch/extension.h>
#include <cuda_runtime.h>

// Custom CUDA kernel for 2D convolution with stride, padding, dilation
__global__ void conv2d_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int batch_size,
    int in_channels,
    int in_h,
    int in_w,
    int out_channels,
    int k_h,
    int k_w,
    int stride,
    int padding,
    int dilation,
    int out_h,
    int out_w) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * out_channels * out_h * out_w;
    if (idx < total) {
        // Decode output indices
        int w_out = idx % out_w;
        int temp = idx / out_w;
        int h_out = temp % out_h;
        temp / out_h;
        int c_out = temp % out_channels;
        
        float sum = bias[c_out];
        
        // Loop over input channels and kernel elements
        for (int c_in = 0; c_in < in_channels; c_in++) {
            for (int kh = 0; kh < k_h; kh++) {
                for (int kw = 0; kw < k_w; kw++) {
                    int in_h = h_out + padding - kh * dilation;
                    int in_w = w_out + padding - kw * dilation;
                    
                    if (in_h >= 0 && in_h < in_h && in_w >= 0 && in_w < in_w) {
                        int input_idx = c_in * (in_h * in_w) + in_h * in_w + in_w;
                        int weight_idx = c_in * (k_h * k_w) + kh * k_w + kw;
                        
                        sum += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }
        output[idx] = sum;
    }
}

torch::Tensor conv2d_cuda(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, int64_t stride, int64_t padding, int64_t dilation) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "Weight must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "Bias must be a CUDA tensor");
    TORCH_CHECK(x.dim() == 4, "Input must be 4D");
    TORCH_CHECK(weight.dim() == 4, "Weight must be 4D");
    TORCH_CHECK(weight.size(1) == x.size(1), "Input channels must match weight channels");

    int batch_size = x.size(0);
    int in_channels = x.size(1);
    int in_h = x.size(2);
    int in_w = x.size(3);
    int out_channels = weight.size(1);
    int k_h = weight.size(2);
    int k_w = weight.size(3);

    int out_h = (in_h - 1) * stride - 2 * padding + k_h * dilation;
    int out_w = (in_w - 1) * stride - 2 * padding + k_w * dilation;

    auto output = torch::empty({batch_size, out_channels, out_h, out_w}, x.options());

    int total = batch_size * out_channels * out_h * out_w;
    int block_size = 256;
    int num_blocks = (total + block_size - 1) / block_size;

    conv2d_kernel<<<num_blocks, block_size>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        in_h,
        in_w,
        out_channels,
        k_h,
        k_w,
        stride,
        padding,
        dilation,
        out_h,
        out_w
    );

    return output;
}"""

cpp_source = """torch::Tensor conv2d_cuda(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, int64_t stride, int64_t padding, int64_t dilation);"""

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
