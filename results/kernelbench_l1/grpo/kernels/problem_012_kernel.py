import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = """#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Custom CUDA kernel for diagonal matmul with optimized thread and block indexing
__global__ void diag_matmul_kernel(const float* A, const float* B, float* C, int N, int M) {
    // Each block processes one row of the diagonal matrix
    int row = blockIdx.x;
    // Each thread processes multiple elements in the column
    int col = threadIdx.x;
    int stride = blockDim.x;
    
    for (int i = col; i < M; i += stride) {
        // Compute diagonal element for this row
        float a_val = A[row];
        // Multiply and store result
        C[row * M + i] = a_val * B[row * M + i];
    }
}

// The forward function exposed via PyBind11
torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.dim() == 1, "A must be a 1D tensor");
    TORCH_CHECK(B.dim() == 2, "B must be a 2D tensor");
    TORCH_CHECK(A.size(0) == B.size(0), "A and B must have the same number of rows");
    TORCH_CHECK(A.size(0) == B.size(1), "A.size(0) must match B.size(1) for column dimension");

    int N = A.size(0);
    int M = B.size(1);
    auto C = torch::empty_like(B);

    // Launch one block per row with a fixed number of threads per block
    const int threads = 256;
    diag_matmul_kernel<<<N, threads>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), N, M);
    
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "Kernel failed: ", cudaGetErrorString(err));

    return C;
}"""

cpp_source = """torch::Tensor forward(torch::Tensor A, torch::Tensor B);"""

custom_ops = load_inline(
    name="custom_ops",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['forward'],
    verbose=False,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"],
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, A, B):
        return custom_ops.forward(A, B)
