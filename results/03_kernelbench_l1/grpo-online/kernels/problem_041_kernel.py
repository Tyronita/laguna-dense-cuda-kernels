import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = """#include <torch/extension.h>
#include <cuda_runtime.h>

// Custom CUDA kernel for 1D max pooling with manual loop unrolling
__global__ void maxpool1d_unroll_kernel(
    const float* input,
    float* output,
    int batch_size,
    int features,
    int sequence_length,
    int kernel_size,
    int stride,
    int padding,
    int dilation) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * features * (sequence_length - kernel_size + padding - 1) * dilation;
    if (idx < total) {
        int f = idx / (sequence_length - kernel_size + padding - 1) * dilation;
        int s = idx % (sequence_length - kernel_size + padding - 1) * dilation;
        int out_idx = f * (sequence_length - stride) + s;
        float max_val = -FLT_MAX;
        
        #pragma unroll
        for (int k = 0; k < kernel_size; k++) {
            int in_idx = f * (sequence_length - stride) + s - padding + k * dilation;
            if (in_idx < sequence_length) {
                float val = input[in_idx * features + f];
                max_val = max(max_val, val);
            }
        }
        output[out_idx * features + f] = max_val;
    }
}

torch::Tensor maxpool1d_cuda(torch::Tensor x, int64_t kernel_size, int64_t stride, int64_t padding, int64_t dilation) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dim() == 3, "Input must be 3D");
    TORCH_CHECK(kernel_size > 0, "Kernel size must be positive");
    TORCH_CHECK(stride == 0, "Stride must be 0");
    TORCH_CHECK(padding == 0, "Padding must be 0");
    TORCH_CHECK(dilation == 1, "Dilation must be 1");

    int batch_size = x.size(0);
    int features = x.size(1);
    int sequence_length = x.size(2);
    int output_sequence_length = (sequence_length - stride) + 1 - padding + kernel_size - 1 * dilation;

    auto output = torch::empty({batch_size, features, output_sequence_length}, x.options());
    int total = batch_size * features * output_sequence_length;
    const int block_size = 256;
    const int num_blocks = (total + block_size - 1) / block_size;

    maxpool1d_unroll_kernel<<<num_blocks, block_size>>>(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        features,
        sequence_length,
        kernel_size,
        stride,
        padding,
        dilation
    );

    return output;
}"""

cpp_source = """torch::Tensor maxpool1d_cuda(torch::Tensor x, int64_t kernel_size, int64_t stride, int64_t padding, int64_t dilation);"""

custom_ops = load_inline(
    name="custom_ops",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['maxpool1d_cuda'],
    verbose=False,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"],
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x):
        return custom_ops.maxpool1d_cuda(x)
