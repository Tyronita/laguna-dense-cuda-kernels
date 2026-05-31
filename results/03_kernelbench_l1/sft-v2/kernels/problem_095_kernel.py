import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = """#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>

__global__ void fusedAddReluKernel(const float* a, const float* b, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = a[idx] + b[idx] > 0.0f ? a[idx] + b[idx] : 0.0f;
    }
}

torch::Tensor fusedAddRelu_cuda(torch::Tensor a, torch::Tensor b) {
    auto size = a.numel();
    auto out = torch::zeros_like(a);
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;
    fusedAddReluKernel<<<num_blocks, block_size>>>(a.data_ptr<float>(), b.data_ptr<float>(), out.data_ptr<float>(), size);
    return out;
}

torch::Tensor cross_entropy_cuda(torch::Tensor predictions, torch::Tensor targets) {
    auto log = torch::logf(predictions);
    auto loss = torch::sum(log * targets, 1) / torch::sqrt(torch::sum(log, 1) + 1e-5);
    return loss;
}

torch::Tensor forward(torch::Tensor a, torch::Tensor b) {
    return fusedAddRelu_cuda(a, b);
}"""

cpp_source = """torch::Tensor fusedAddRelu_cuda(torch::Tensor a, torch::Tensor b);\ntorch::Tensor cross_entropy_cuda(torch::Tensor predictions, torch::Tensor targets);\ntorch::Tensor forward(torch::Tensor a, torch::Tensor b);"""

custom_ops = load_inline(
    name="custom_ops",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['fusedAddRelu_cuda', 'cross_entropy_cuda', 'forward'],
    verbose=False,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"],
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, predictions, targets):
        return custom_ops.fusedAddRelu_cuda(predictions, targets)
