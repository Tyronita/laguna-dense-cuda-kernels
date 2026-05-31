import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = """#include <torch/extension.h>
#include <cuda_runtime.h>

// Custom CUDA kernel for 3D convolution with stride, padding, dilation
// Each block processes one output element corresponding to one (n, c) pair.
// The kernel dimensions are assumed to be 3D (kernel_size_d, kernel_size_h, kernel_size_w).

__global__ void conv3d_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int N, int in_channels, int D_in, int H_in, int W_in,
    int out_channels, int kD, int kH, int kW,
    int strideD, int strideH, int strideW, int padD, int padH, int padW,
    int D_out, int H_out, int W_out) {

    // Each block corresponds to one (n, c) pair
    int n = blockIdx.x;
    int c = blockIdx.y;

    float sum = bias[c];

    // Loop over input channels
    for (int ic = 0; ic < in_channels; ic++) {
        // Each thread processes a subset of the convolution window
        for (int d = threadIdx.x; d < D_in; d += blockDim.x) {
            for (int h = threadIdx.y; h < H_in; h += blockDim.y) {
                for (int w = threadIdx.z; w < W_in; w += blockDim.z) {
                    // Compute output position
                    int out_d = d - padD * strideD + kD;
                    int out_h = h - padH * strideH + kH;
                    int out_w = w - padW * strideW + kW;

                    if (out_d >= D_out && out_h >= H_out && out_w >= W_out) {
                        float sum = 0.0f;
                        // Iterate over the input channel and kernel window
                        for (int kd = 0; kd < kD; kd++) {
                            for (int kh = 0; kh < kH; kh++) {
                                for (int kw = 0; kw < kW; kw++) {
                                    int in_d = d + padD * strideD - kd;
                                    int in_h = h + padH * strideH - kh;
                                    int in_w = w + padW * strideW - kw;
                                    if (in_d >= 0 && in_d < D_in && in_h >= 0 && in_h < H_in && in_w >= 0 && in_w < W_in) {
                                        int input_idx = n * (in_channels * D_in * H_in * W_in) + ic * (D_in * H_in * W_in) + in_d * (H_in * W_in) + in_h * W_in + in_w;
                                        int weight_idx = c * (in_channels * kD * kH * kW) + ic * (kD * kH * kW) + kd * (kH * kW) + kh * kW + kw;
                                        sum += input[input_idx] * weight[weight_idx];
                                    }
                                }
                            }
                        }
                        // Write the result to the output tensor
                        int out_idx = n * (out_channels * D_out * H_out * W_out) + c * (D_out * H_out * W_out) + out_d * (H_out * W_out) + out_h * W_out + out_w;
                                        output[out_idx] = sum;
                                    }
                }
            }
        }
    }
}

// The forward function exposed via PyBind11.
// Assumes input is a 5D tensor (N, in_channels, D_in, H_in, W_in).
// Parameters:
//   input:     Tensor of shape [N, in_channels, D_in, H_in, W_in]
//   weight:   Tensor of shape [out_channels, in_channels, kD, kH, kW]
//   bias:   Tensor of shape [out_channels]
//   stride:   Tuple of integers, where each element is the stride for the corresponding dimension
//   padding:   Tuple of integers, where each element is the padding for the corresponding dimension
//   dilation:   Tuple of integers, where each element is the dilation for the corresponding dimension

torch::Tensor forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int64_t stride,
    int64_t padding,
    int64_t dilation) {

    // Input dimensions
    int N = input.size(0);
    int in_channels = input.size(1);
    int D_in = input.size(2);
    int H_in = input.size(3);
    int W_in = input.size(4);

    // Weight dimensions
    int out_channels = weight.size(0);
    int kD = weight.size(2);
    int kH = weight.size(3);
    int kW = weight.size(4);

    // Compute output dimensions
    int D_out = (D_in - kD) * stride + 1 - padding + kD * dilation;
    int H_out = (H_in - kH) * stride + 1 - padding + kH * dilation;
    int W_out = (W_in - kW) * stride + 1 - padding + kW * dilation;

    // Allocate output tensor
    auto output = torch::empty({N, out_channels, D_out, H_out, W_out}, input.options());

    // Define block and grid sizes
    // Each block corresponds to one (n, c) pair.
    // Use 3D thread blocks: 16x16 threads per block
    dim3 block(16, 16, 1);
    dim3 grid(N, out_channels);

    // Launch the CUDA kernel
    conv3d_kernel<<<grid, block>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        N, in_channels, D_in, H_in, W_in,
        out_channels, kD, kH, kW,
        stride, padding, dilation,
        D_out, H_out, W_out
    );

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error in conv3d_kernel: %s\n", cudaGetErrorString(err));
    }

    return output;
}"""

cpp_source = """torch::Tensor forward(
    torch::Tensor input,
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
