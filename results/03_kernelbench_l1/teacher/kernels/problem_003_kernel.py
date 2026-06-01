import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

batched_bmm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void batched_bmm_kernel(const float* A, const float* B, float* C, int batch_size, int m, int k, int n) {
    int batch_idx = blockIdx.z;
    int row_idx = blockIdx.y;
    int col_idx = blockIdx.x;
    
    if (batch_idx < batch_size && row_idx < m && col_idx < n) {
        int c_idx = batch_idx * m * n + row_idx * n + col_idx;
        float sum = 0.0f;
        for (int i = 0; i < k; i++) {
            int a_idx = batch_idx * m * k + row_idx * k + i;
            int b_idx = batch_idx * k * n + i * n + col_idx;
            sum += A[a_idx] * B[b_idx];
        }
        C[c_idx] = sum;
    }
}

torch::Tensor batched_bmm_cuda(torch::Tensor A, torch::Tensor B) {
    auto batch_size = A.size(0);
    auto m = A.size(1);
    auto k = A.size(2);
    auto n = B.size(2);
    
    auto C = torch::zeros({batch_size, m, n}, A.options());
    
    const int block_size_x = 16;
    const int block_size_y = 16;
    const int block_size_z = 1;
    
    dim3 block(block_size_x, block_size_y, block_size_z);
    dim3 grid((n + block_size_x - 1) / block_size_x, (m + block_size_y - 1) / block_size_y, batch_size);
    
    batched_bmm_kernel<<<grid, block>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        batch_size, m, k, n
    );
    
    return C;
}
"""

batched_bmm_cpp_source = "torch::Tensor batched_bmm_cuda(torch::Tensor A, torch::Tensor B);"

batched_bmm = load_inline(
    name="batched_bmm",
    cpp_sources=batched_bmm_cpp_source,
    cuda_sources=batched_bmm_source,
    functions=["batched_bmm_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.batched_bmm = batched_bmm

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return self.batched_bmm.batched_bmm_cuda(A, B)