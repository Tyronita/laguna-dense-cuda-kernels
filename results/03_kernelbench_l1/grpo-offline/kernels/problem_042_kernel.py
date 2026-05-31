import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = """#include <torch/extension.h>
#include <cuda_runtime.h>

// Custom CUDA kernel for 2D max pooling with loop unrolling
__global__ void maxpool2d_unroll_kernel(
    const float* input,
    float* output,
    int N, int C, int H, int W,
    int kH, int kW,
    int stride, int padding, int dilation) {

    int outH = (H - 1) * stride - 2 * padding + kH * dilation;
    int outW = (W - 1) * stride - 2 * padding + kW * dilation;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * outH * outW;
    if (idx < total) {
        int c = (idx / (outH * outW)) % C;
        int h = (idx / outW) % outH;
        int w = idx % outW;

        float max_val = -FLT_MAX;
        int base = c * (H * W) + h * W + w;

        #pragma unroll
        for (int kh = 0; kh < kH; ++kh) {
            int inH = h * stride + kh * dilation;
            #pragma unroll
            for (int kw = 0; kw < kW; ++kw) {
                int inW = w * stride + kw * dilation;
                if (inH >= 0 && inH < H && inW >= 0 && inW < W) {
                    float val = input[base + inH * W + inW];
                    max_val = max(max_val, val);
                }
            }
        }
        output[idx] = max_val;
    }
}

torch::Tensor maxpool2d_cuda(torch::Tensor x, int64_t kernel_size, int64_t stride, int64_t padding, int64_t dilation) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dim() == 4, "Input must be 4D");
    TORCH_CHECK(kernel_size % 2 == 0, "Kernel size must be 2D");
    TORCH_CHECK(dilation % 2 == 0, "Dilation must be 2D");

    int N = x.size(0), C = x.size(1), H = x.size(2), W = x.size(3);
    int kH = kernel_size / 2, kW = kernel_size % 2;
    int outH = (H - 1) * stride - 2 * padding + kH * dilation;
    int outW = (W - 1) * stride - 2 * padding + kW * dilation;

    auto output = torch::empty({N, C, outH, outW}, x.options());
    int total = N * C * outH * outW;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    maxpool2d_unroll_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, H, W,
        kH, kW,
        stride, padding,
        dilation
    );

    return output;
}"""

cpp_source = """torch::Tensor maxpool2d_cuda(torch::Tensor x, int64_t kernel_size, int64_t stride, int64_t padding, int64_t dilation);"""

custom_ops = load_inline(
    name="custom_ops",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['maxpool2d_cuda'],
    verbose=False,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"],
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x):
        return custom_ops.maxpool2d_cuda(x)
