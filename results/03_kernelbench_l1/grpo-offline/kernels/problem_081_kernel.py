import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = """#include <torch/extension.h>
#include <cuda_runtime.h>

// Custom CUDA kernel for 2D transposed convolution with stride, padding, and dilation
// This kernel assumes that the input tensor x has shape [N, in_channels, H_in, W_in]
// and the output tensor y has shape [N, out_channels, H_out, W_out].
// The convolution weights are assumed to be in shape [out_channels, in_channels, kH, kW].

__global__ void conv_transpose2d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int N, int in_channels, int H_in, int W_in,
    int out_channels, int kH, int kW,
    int stride, int padding, int dilation,
    int H_out, int W_out) {

    // Compute output dimensions
    int h_out = (H_in - 1) * stride - 2 * padding + kH * dilation;
    int w_out = (W_in - 1) * stride - 2 * padding + kW * dilation;

    // Compute global index for output tensor
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * out_channels * H_out * W_out;
    if (idx < total) {
        // Decode output index into (n, oc, h_out, w_out)
        int w_out_idx = idx % W_out;
        int temp = idx / W_out;
        int h_out_idx = temp % H_out;
        temp /= H_out;
        int oc = temp % out_channels;
        int n = temp / out_channels;

        float sum = bias[oc];

        // Loop over input channels and kernel spatial dimensions
        for (int c = 0; c < in_channels; c++) {
            for (int kh = 0; kh < kH; kh++) {
                for (int kw = 0; kw < kW; kw++) {
                    // Compute corresponding input indices
                    int h_in = h_out_idx + padding - kh * dilation;
                    int w_in = w_out_idx + padding - kw * dilation;
                    if (h_in >= 0 && h_in < H_in && w_in >= 0 && w_in < W_in) {
                        // Input index: [n, c, h_in, w_in]
                        int input_idx = n * (in_channels * H_in * W_in) + c * (H_in * W_in) + h_in * W_in + w_in;
                        // Weight index: [oc, c, kh, kw]
                        int weight_idx = oc * (in_channels * kH * kW) + c * (kH * kW) + kh * kW + kw;
                        sum += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }
        output[idx] = sum;
    }
}

// The forward function exposed via PyBind11.
torch::Tensor forward(
    torch::Tensor x,
    int64_t stride,
    int64_t padding,
    int64_t dilation,
    bool is_training,
    torch::Tensor conv_transpose2d,
    torch::Tensor conv_transpose2d_bias) {

    // Ensure input is contiguous and on CUDA
    x = x.contiguous().cuda();
    // Ensure weights and bias are contiguous and on CUDA
    conv_transpose2d = conv_transpose2d.contiguous().cuda();
    if (is_training) {
        conv_transpose2d_bias = conv_transpose2d_bias.contiguous().cuda();
    } else {
        conv_transpose2d_bias = torch::empty_like(conv_transpose2d);
    }

    // Get input dimensions
    int N = x.size(0);
    int in_channels = x.size(1);
    int H_in = x.size(2);
    int W_in = x.size(3);

    // Get output dimensions
    int out_channels = conv_transpose2d.size(1);
    int kH = conv_transpose2d.size(2);
    int kW = conv_transpose2d.size(3);

    // Compute output spatial dimensions
    int H_out = (H_in - 1) * stride - 2 * padding + kH * dilation;
    int W_out = (W_in - 1) * stride - 2 * padding + kW * dilation;

    // Allocate output tensor
    auto output = torch::empty({N, out_channels, H_out, W_out}, x.options());

    // Define block and grid sizes
    int threads = 256;
    int total = N * out_channels * H_out * W_out;
    int blocks = (total + threads - 1) / threads;

    // Launch the CUDA kernel
    conv_transpose2d_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        conv_transpose2d.data_ptr<float>(),
        conv_transpose2d_bias.data_ptr<float>(),
        output.data_ptr<float>(),
        N, in_channels, H_in, W_in,
        out_channels, kH, kW,
        stride, padding, dilation,
        H_out, W_out
    );

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error in conv_transpose2d_kernel: %s\n", cudaGetErrorString(err));
    }

    return output;
}"""

cpp_source = """torch::Tensor forward(
    torch::Tensor x,
    int64_t stride,
    int64_t padding,
    int64_t dilation,
    bool is_training,
    torch::Tensor conv_transpose2d,
    torch::Tensor conv_transpose2d_bias);"""

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
