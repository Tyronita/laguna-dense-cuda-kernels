import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = """#include <torch/extension.h>
#include <cuda_runtime.h>

// Custom CUDA kernel for matrix multiplication with optimized thread and block indexing
__global__ void matmul_kernel(const float* A, const float* B, float* C, int N) {
    // Calculate global thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    // Loop over elements with stride
    for (int i = idx; i < N; i += stride) {
        float sum = 0.0f;
        // Compute dot product for this element
        for (int k = 0; k < N; k++) {
            sum += A[i * N + k] * B[k * N + i];
        }
        C[i * N + i] = sum;
    }
}

// C++ interface exposed via PyBind11
torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "Input A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "Input B must be a CUDA tensor");
    TORCH_CHECK(A.dim() == 2, "Input A must be a 2D tensor");
    TORCH_CHECK(B.dim() == 2, "Input B must be a 2D tensor");
    TORCH_CHECK(A.size(0) == A.size(1), "Input A must be square");
    TORCH_CHECK(B.size(0) == B.size(1), "Input B must be square");
    TORCH_CHECK(A.size(0) == B.size(0), "Input A and B must be the same size");

    int N = A.size(0);
    auto C = torch::zeros_like(A);

    // Optimize thread and block configuration
    const int threads = 256;
    const int blocks = (N + threads - 1) / threads;

    matmul_kernel<<<blocks, threads>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), N);
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
