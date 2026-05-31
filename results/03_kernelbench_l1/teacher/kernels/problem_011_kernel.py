import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

matmul_4d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void matmul_4d_kernel(const float* A, const float* B, float* C, int b, int i, int j, int l, int k) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = b * i * j * k;
    if (idx < total) {
        int k_idx = idx % k;
        int j_idx = (idx / k) % j;
        int i_idx = (idx / (k * j)) % i;
        int b_idx = idx / (k * j * i);
        
        float sum = 0;
        for (int ll = 0; ll < l; ll++) {
            int a_idx = ((b_idx * i + i_idx) * j + j_idx) * l + ll;
            int b_idx2 = ll * k + k_idx;
            sum += A[a_idx] * B[b_idx2];
        }
        int c_idx = ((b_idx * i + i_idx) * j + j_idx) * k + k_idx;
        C[c_idx] = sum;
    }
}

torch::Tensor matmul_4d_cuda(torch::Tensor A, torch::Tensor B) {
    auto b = A.size(0);
    auto i = A.size(1);
    auto j = A.size(2);
    auto l = A.size(3);
    auto k = B.size(1);
    
    auto C = torch::zeros({b, i, j, k}, A.options());
    const int block_size = 256;
    const int num_blocks = (b * i * j * k + block_size - 1) / block_size;
    matmul_4d_kernel<<<num_blocks, block_size>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), b, i, j, l, k);
    return C;
}
"""
matmul_4d_cpp_source = "torch::Tensor matmul_4d_cuda(torch::Tensor A, torch::Tensor B);"
matmul_4d = load_inline(name="matmul_4d", cpp_sources=matmul_4d_cpp_source, cuda_sources=matmul_4d_source, functions=["matmul_4d_cuda"], verbose=True)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.matmul_4d = matmul_4d

    def forward(self, A, B):
        return self.matmul_4d.matmul_4d_cuda(A, B)