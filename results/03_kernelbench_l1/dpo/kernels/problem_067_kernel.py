import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = """#include <torch/extension.h>
#include <cuda_runtime.h>

// Custom CUDA kernel for 1D convolution with optimized block size
__global__ void conv1d_kernel(
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
    int padding) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * out_channels * in_h * in_w;
    if (idx < total) {
        // Decode output indices
        int w_out = idx % in_w;
        int temp = idx / in_w;
        int h_out = temp % in_h;
        temp / in_h;
        int c_out = temp % out_channels;
        
        float sum = bias[c_out];
        
        // Compute convolution with stride and padding
        for (int c_in = 0; c_in < in_channels; c_in++) {
            for (int kh = 0; kh < k_h; kh++) {
                for (int kw = 0; kw < k_w; kw++) {
                    int h_in = h_out + padding - kh;
                    int w_in = w_out + padding - kw;
                    
                    if (h_in >= 0 && h_in < in_h && w_in >= 0 && w_in < in_w) {
                        int input_idx = c_in * (in_h * in_w) + h_in * in_w + w_in;
                        int weight_idx = c_in * (out_channels * k_h * k_w) + c_out * (k_h * k_w) + kh * k_w + kw;
                        sum += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }
        output[idx] = sum;
    }
}

torch::Tensor conv1d_cuda(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, int64_t stride, int64_t padding) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "Weight must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "Bias must be a CUDA tensor");
    TORCH_CHECK(x.dim() == 3, "Input must be 3D");
    TORCH_CHECK(weight.dim() == 3, "Weight must be 3D");
    TORCH_CHECK(weight.size(1) == x.size(1), "Input channels must match weight channels");
    TORCH_CHECK(bias.dim() == 1, "Bias must be 1D");
    TORCH_CHECK(bias.size(1) == weight.size(1), "Bias size must match weight channels");

    int batch_size = x.size(0);
    int in_channels = x.size(1);
    int in_h = x.size(2);
    int in_w = x.size(3);
    int out_channels = weight.size(1);
    int k_h = weight.size(2);
    int k_w = weight.size(3);

    int out_h = (in_h - 1) * stride - 2 * padding + k_h;
    int out_w = (in_w - 1) * stride - 2 * padding + k_w;
    TORCH_CHECK(out_h > 0 && out_w > 0, "Output dimensions must be positive");

    auto output = torch::empty({batch_size, out_channels, out_h, out_w}, x.options());
    int total = batch_size * out_channels * out_h * out_w;
    const int block_size = 256;
    const int num_blocks = (total + block_size - 1) / block_size;

    conv1d_kernel<<<num_blocks, block_size>>>(
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
        padding
    );

    return output;
}"""

cpp_source = """torch::Tensor conv1d_cuda(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, int64_t stride, int64_t padding);"""

custom_ops = load_inline(
    name="custom_ops",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['conv1d_cuda'],
    verbose=False,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"],
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x):
        return custom_ops.conv1d_cuda(x)
