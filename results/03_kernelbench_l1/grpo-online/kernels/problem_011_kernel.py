import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = """#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void einsum_kernel(const float* A, const float* B, float* C, int BATCH, int I, int J, int L, int K) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = BATCH * I * J * K;
    if (idx < total) {
        int k = idx % K;
        int remainder = idx / K;
        int j = remainder % J;
        remainder /= J;
        int i = remainder % I;
        int b = remainder / I;
        
        float sum = 0.0f;
        for (int l = 0; l < L; ++l) {
            int a_idx = b * (I * J * L) + i * (J * L) + j * L + l;
            int b_idx = l * K + k;
            sum += A[a_idx] * B[b_idx];
        }
        C[idx] = sum;
    }
}

torch::Tensor einsum_cuda(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.dim() == 4, "A must be 4D");
    TORCH_CHECK(B.dim() == 2, "B must be 2D");
    TORCH_CHECK(A.size(3) == B.size(0), "A.size(3) must match B.size(0)");

    int BATCH = A.size(0), I = A.size(1), J = A.size(2), L = A.size(3);
    int K = B.size(1);
    
    auto C = torch::empty({BATCH, I, J, K}, A.options());
    
    const int block_size = 256;
    const int num_blocks = (BATCH * I * J * K + block_size - 1) / block_size;
    
    einsum_kernel<<<num_blocks, block_size>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), 
        BATCH, I, J, L, K
    );
    
    return C;
}"""

cpp_source = """torch::Tensor einsum_cuda(torch::Tensor A, torch::Tensor B);"""

custom_ops = load_inline(
    name="custom_ops",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['einsum_cuda'],
    verbose=False,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"],
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, A, B):
        return custom_ops.einsum_cuda(A, B)
