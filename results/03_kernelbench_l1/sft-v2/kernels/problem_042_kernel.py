import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = """#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void maxpool2d_kernel(
    const float* input,
    float* output,
    int N, int C, int H, int W,
    int kH, int kW,
    int stride, int padding, int dilation) {

    int outH = (H - padding + kH - 1) / stride + 1;
    int outW = (W - padding + kW - 1) / stride + 1;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * outH * outW;
    if (idx < total) {
        int c = (idx / (outH * outW)) % C;
        int h = (idx / outW) % outH;
        int w = idx % outW;
        float max_val = -FLT_MAX;
        for (int i = 0; i < kH; i++) {
            for (int j = 0; j < kW; j++) {
                int inH = h * stride + i - padding;
                int inW = w * stride + j - padding;
                if (inH >= 0 && inH < H && inW >= 0 && inW < W) {
                    int index = c * (H * W) + inH * W + inW;
                    float val = input[index];
                    max_val = max(max_val, val);
                }
            }
        }
        output[idx] = max_val;
    }
}

torch::Tensor maxpool2d_cuda(torch::Tensor x, int64_t kernel_size, int64_t stride, int64_t padding, int64_t dilation) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");

    int N = x.size(0), C = x.size(1), H = x.size(2), W = x.size(3);
    int kH = kernel_size / 2, kW = kernel_size / 2;
    int outH = (H - padding + kH - 1) / stride + 1;
    int outW = (W - padding + kW - 1) / stride + 1;

    auto output = torch::empty({N, C, outH, outW}, x.options());
    int total = N * C * outH * outW;
    const int block_size = 256;
    const int num_blocks = (total + block_size - 1) / block_size;

    maxpool2d_kernel<<<num_blocks, block_size>>>(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, H, W,
        kH, kW,
        stride, padding, dilation
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
