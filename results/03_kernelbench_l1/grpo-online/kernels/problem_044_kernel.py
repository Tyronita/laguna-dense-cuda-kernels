import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = """#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void avg_pool1d_kernel(const float* input, float* output,
                                   int batch_size, int in_channels, int input_length,
                                   int kernel_size, int stride, int padding) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * in_channels * (input_length - kernel_size + padding) * stride;
    if (idx < total) {
        int c = idx / (input_length - kernel_size + padding) * stride;
        int h = (idx % (input_length - kernel_size + padding)) * stride;
        float sum = 0.0f;
        for (int i = 0; i < kernel_size; i++) {
            int in_idx = h + i - padding;
            if (in_idx >= input_length) {
                int input_idx = c * in_channels * input_length + in_idx;
                sum += input[input_idx];
            }
        }
        output[idx] = sum / kernel_size;
    }
}

torch::Tensor avg_pool1d_cuda(torch::Tensor x, int64_t kernel_size, int64_t stride, int64_t padding) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be a CUDA tensor");
    TORCH_CHECK(x.dim() == 3, "Input tensor must be 3D");
    TORCH_CHECK(kernel_size > 0, "Kernel size must be positive");
    TORCH_CHECK(stride > 0, "Stride must be positive");
    TORCH_CHECK(padding > 0, "Padding must be positive");

    int batch_size = x.size(0);
    int in_channels = x.size(1);
    int input_length = x.size(2);
    int output_length = (input_length - kernel_size + padding) * stride;
    auto output = torch::empty({batch_size, in_channels, output_length}, x.options());

    int total = batch_size * in_channels * output_length;
    const int block_size = 256;
    const int num_blocks = (total + block_size - 1) / block_size;

    avg_pool1d_kernel<<<num_blocks, block_size>>>(
        x.data_ptr<float>(), output.data_ptr<float>(),
        batch_size, in_channels, input_length,
        kernel_size, stride, padding
    );

    return output;
}"""

cpp_source = """torch::Tensor avg_pool1d_cuda(torch::Tensor x, int64_t kernel_size, int64_t stride, int64_t padding);"""

custom_ops = load_inline(
    name="custom_ops",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['avg_pool1d_cuda'],
    verbose=False,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"],
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x):
        return custom_ops.avg_pool1d_cuda(x)
