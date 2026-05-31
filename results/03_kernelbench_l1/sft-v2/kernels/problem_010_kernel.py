import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = """#include <torch/extension.h>
#include <cuda_runtime.h>

// Custom CUDA kernel for 3D tensor-matrix multiplication with loop unrolling
__global__ void matmul_unroll_kernel(const float* A, const float* B, float* C, int N, int M, int K, int L) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * M * L;
    if (idx < total) {
        int n = idx / (M * L);
        int m = (idx / L) % M;
        int l = idx % L;
        float sum = 0.0f;
        #pragma unroll 4
        for (int k = 0; k < K; k++) {
            sum += A[n * M * K + m * K + k] * B[k * L + l];
        }
        C[n * M * L + m * L + l] = sum;
    }
}

// C++ interface exposed via PyBind11
torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.dim() == 3, "A must be 3D");
    TORCH_CHECK(B.dim() == 2, "B must be 2D");
    TORCH_CHECK(A.size(2) == B.size(0), "A.size(2) must match B.size(0)");

    int N = A.size(0), M = A.size(1), K = A.size(2);
    int L = B.size(1);

    auto C = torch::zeros({N, M, L}, A.options());
    int total = N * M * L;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    matmul_unroll_kernel<<<blocks, threads>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), N, M, K, L
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "Kernel failed: ", cudaGetErrorString(err));

    return C;
}"""

cpp_source = """torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B);"""

custom_ops = load_inline(
    name="custom_ops",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['matmul_cuda'],
    verbose=False,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"],
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, A, B):
        return custom_ops.matmul_cuda(A, B)
