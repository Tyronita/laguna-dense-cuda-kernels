import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = """#include <torch/extension.h>
#include <cuda_runtime.h>

// Custom CUDA kernel for batched matrix multiplication with loop unrolling
__global__ void bmm_unroll_kernel(const float* A, const float* B, float* C, int batch_size, int M, int K, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * M * N;
    if (idx < total) {
        int m = idx / N;
        int n = idx % N;
        int b = idx / (M * N);
        float sum = 0.0f;
        #pragma unroll 4
        for (int k = 0; k < K; k++) {
            sum += A[b * M * K + m * K + k] * B[b * K * N + k * N + n];
        }
        C[b * M * N + m * N + n] = sum;
    }
}

// C++ interface exposed via PyBind11
torch::Tensor bmm_cuda(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "Input A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "Input B must be a CUDA tensor");
    TORCH_CHECK(A.dim() == 3, "Input A must be 3D");
    TORCH_CHECK(B.dim() == 3, "Input B must be 3D");
    TORCH_CHECK(A.size(0) == B.size(0), "Batch sizes must match");
    TORCH_CHECK(A.size(2) == B.size(1), "Inner dimensions (K) must match");

    int batch_size = A.size(0);
    int M = A.size(1);
    int K = A.size(2);
    int N = B.size(2);

    auto C = torch::zeros({batch_size, M, N}, A.options());
    int total = batch_size * M * N;
    const int threads = 256;
    const int blocks = (total + threads - 1) / threads;

    bmm_unroll_kernel<<<blocks, threads>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(),
        batch_size, M, K, N
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "Kernel failed: ", cudaGetErrorString(err));

    return C;
}"""

cpp_source = """torch::Tensor bmm_cuda(torch::Tensor A, torch::Tensor B);"""

custom_ops = load_inline(
    name="custom_ops",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['bmm_cuda'],
    verbose=False,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"],
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, A, B):
        return custom_ops.bmm_cuda(A, B)
