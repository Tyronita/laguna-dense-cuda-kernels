import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = """#include <torch/extension.h>
#include <cuda_runtime.h>

// Custom CUDA kernel for 3D max pooling with manual loop unrolling
__global__ void maxpool3d_unroll_kernel(
    const float* input,
    float* output,
    int N, int C, int D, int D2, int D3,
    int kD, int kD2, int kD3,
    int stride, int padding, int dilation) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * (D / stride) * (D2 / stride) * (D3 / stride);
    if (idx < total) {
        int c = idx / (D * D2 * D3);
        int rem = idx % (D * D2 * D3);
        int d1 = rem / (D2 * D3);
        int d2 = rem % (D2 * D3);
        int d3 = d1 / D2;

        float max_val = -FLT_MAX;
        int d_start = d1 * stride - padding;
        int d_end = d2 * stride - padding;
        int d_end3 = d3 * stride - padding;

        #pragma unroll
        for (int d0 = 0; d0 < kD; d++) {
            int d0_offset = d_start + d0 - padding;
            #pragma unroll
            for (int d1 = 0; d1 < kD; d1++) {
                int d1_offset = d_end + d1 - padding;
                #pragma unroll
                for (int d2 = 0; d2 < kD; d2++) {
                    int d2_offset = d_end3 + d2 - padding;
                    int input_idx = c * (D * D2 * D3) + d0_offset * (D2 * D3) + d1_offset * D3 + d2_offset;
                    int input_val = input[input_idx];
                    max_val = max(max_val, input_val);
                }
            }
        }
        output[idx] = max_val;
    }
}

torch::Tensor maxpool3d_cuda(torch::Tensor x, int64_t kernel_size, int64_t stride, int64_t padding, int64_t dilation) {
    TORCH_CHECK(x.dim() == 5, "Input must be 5D");
    TORCH_CHECK(kernel_size > 0, "Kernel size must be positive");
    TORCH_CHECK(stride == kernel_size, "Stride must match kernel size");
    TORCH_CHECK(dilation == 1, "Dilation must be 1");

    int N = x.size(0), C = x.size(1), D = x.size(2), D2 = x.size(3), D3 = x.size(4);
    int kD = kernel_size, kD2 = kernel_size, kD3 = kernel_size;
    int outD = (D - 1) / stride + 1, outD2 = (D2 - 1) / stride + 1, outD3 = (D3 - 1) / stride + 1;

    auto output = torch::empty({N, C, outD, outD2, outD3}, x.options());
    int total = N * C * outD * outD2 * outD3;
    const int block_size = 256;
    const int num_blocks = (total + block_size - 1) / block_size;

    maxpool3d_unroll_kernel<<<num_blocks, block_size>>>(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, D, D2, D3,
        kD, kD2, kD3,
        stride, padding, dilation
    );

    return output;
}"""

cpp_source = """torch::Tensor maxpool3d_cuda(torch::Tensor x, int64_t kernel_size, int64_t stride, int64_t padding, int64_t dilation);"""

custom_ops = load_inline(
    name="custom_ops",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['maxpool3d_cuda'],
    verbose=False,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"],
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x):
        return custom_ops.maxpool3d_cuda(x)
