import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = """#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void elementwiseMultiplyKernel(const float* A, const float* B, float* C, float s, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        C[idx] = A[idx] * s;
    }
}

torch::Tensor elementwiseMultiply_cuda(torch::Tensor A, float s) {
    auto size = A.numel();
    auto C = torch::empty_like(A);
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;
    elementwiseMultiplyKernel<<<num_blocks, block_size>>>(A.data_ptr<float>(), s, C.data_ptr<float>(), size);
    return C;
}"""

cpp_source = """torch::Tensor elementwiseMultiply_cuda(torch::Tensor A, float s);"""

custom_ops = load_inline(
    name="custom_ops",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['elementwiseMultiply_cuda'],
    verbose=False,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"],
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, A, s):
        return custom_ops.elementwiseMultiply_cuda(A, s)
