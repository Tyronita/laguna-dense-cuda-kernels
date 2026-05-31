import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = """#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void elementwise_add_kernel(const float* a, const float* b, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) { out[idx] = a[idx] + b[idx];
    }
}

torch::Tensor elementwise_add_cuda(torch::Tensor a, torch::Tensor b) {
    auto size = a.numel();
    auto out = torch::zeros_like(a);
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;
    elementwise_add_kernel<<<num_blocks, block_size>>>(a.data_ptr<float>(), b.data_ptr<float>(), out.data_ptr<float>(), size);
    return out;
}

torch::Tensor conv2d_cuda(torch::Tensor x, torch::Tensor weight, torch::Tensor bias) {
    return at::conv2d(x, weight, bias, {1, 1}, {2, 2});
}

torch::Tensor forward(torch::Tensor x, torch::Tensor conv1_weight, torch::Tensor conv1_bias) {
    x = conv2d_cuda(x, conv1_weight, conv1_bias);
    return x;
}"""

cpp_source = """torch::Tensor elementwise_add_cuda(torch::Tensor a, torch::Tensor b);\ntorch::Tensor conv2d_cuda(torch::Tensor x, torch::Tensor weight, torch::Tensor bias);\ntorch::Tensor forward(torch::Tensor x, torch::Tensor conv1_weight, torch::Tensor conv1_bias);"""

custom_ops = load_inline(
    name="custom_ops",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['elementwise_add_cuda', 'conv2d_cuda', 'forward'],
    verbose=False,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"],
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x):
        return custom_ops.elementwise_add_cuda(x)
