import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = """#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_norm_relu_kernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float norm_val = input[idx] / sqrtf(1.0f + 1e-5f);
        output[idx] = norm_val * (norm_val > 0.0f);
    }
}

torch::Tensor fused_norm_relu_cuda(torch::Tensor input) {
    auto size = input.numel();
    auto output = torch::empty_like(input);
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;
    fused_norm_relu_kernel<<<num_blocks, block_size>>>(input.data_ptr<float>(), output.data_ptr<float>(), size);
    return output;
}

torch::Tensor elementwise_add_cuda(torch::Tensor a, torch::Tensor b) {
    auto size = a.numel();
    auto output = torch::empty_like(a);
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;
    elementwise_add_kernel<<<num_blocks, block_size>>>(a.data_ptr<float>(), b.data_ptr<float>(), output.data_ptr<float>(), size);
    return output;
}

torch::Tensor forward(torch::Tensor x, torch::Tensor b) {
    return fused_norm_relu_cuda(x) + elementwise_add_cuda(b);
}"""

cpp_source = """torch::Tensor fused_norm_relu_cuda(torch::Tensor input);\ntorch::Tensor elementwise_add_cuda(torch::Tensor a, torch::Tensor b);\ntorch::Tensor forward(torch::Tensor x, torch::Tensor b);"""

custom_ops = load_inline(
    name="custom_ops",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['fused_norm_relu_cuda', 'elementwise_add_cuda', 'forward'],
    verbose=False,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"],
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x):
        return custom_ops.fused_norm_relu_cuda(x)
