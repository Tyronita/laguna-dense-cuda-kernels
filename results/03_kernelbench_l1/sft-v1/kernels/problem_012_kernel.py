import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = """#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// CUDA kernel for diagonal matrix multiplication with vectorized loads
template <typename scalar_t>
__global__ void diag_matmul_kernel_vectorized(
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ A,
    const scalar_t* __restrict__ B,
    const int64_t N,
    const int64_t M) {
    
    const int row = blockIdx.x;
    const int col = blockIdx.y * blockDim.y + threadIdx.y;
    const int lane_id = threadIdx.x;
    
    if (row < N && col < M) {
        scalar_t sum = 0;
        const scalar_t* A_row = A + row;
        const scalar_t* B_col = B + col;
        
        // Vectorized loop for better memory coalescing
        const int vec_size = 4;
        const int vec_elements = M / vec_size;
        const float4* B_vec = reinterpret_cast<const float4*>(B_col);
        
        for (int i = 0; i < vec_elements; i += vec_size) {
            float4 b_vec = *reinterpret_cast<const float4*>(B_vec)[i];
            sum += A_row.x * b_vec.x;
            sum += A_row.x * b_vec.y;
            sum += A_row.x * b_vec.z;
            sum += A_row.x * b_vec.w;
        }
        
        // Handle remaining elements
        const int remainder_start = vec_elements * vec_size;
        for (int i = remainder_start; i < M; i++) {
            sum += A_row.x * B_col[i];
        }
        
        output[row * M + col] = sum;
    }
}

// PyTorch wrapper function
torch::Tensor diag_matmul_cuda(torch::Tensor A, torch::Tensor B) {
    const int64_t N = A.size(0);
    const int64_t M = B.size(1);
    
    auto options = torch::TensorOptions()
        .dtype(A.dtype())
        .device(A.device());
    auto output = torch::empty({N, M}, options);
    
    const int threads = 16;
    dim3 threads(threads, threads);
    dim3 blocks(
        (N + threads - 1) / threads,
        (M + threads - 1) / threads
    );
    
    AT_DISPATCH_FLOATING_TYPES(A.type(), "diag_matmul_kernel_vectorized", ([&] {
        diag_matmul_kernel_vectorized<scalar_t><<<blocks, threads>>>(
            output.data_ptr<scalar_t>(),
            A.data_ptr<scalar_t>(),
            B.data_ptr<scalar_t>(),
            N,
            M
        );
    }));
    
    return output;
}"""

cpp_source = """torch::Tensor diag_matmul_cuda(torch::Tensor A, torch::Tensor B);"""

custom_ops = load_inline(
    name="custom_ops",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['diag_matmul_cuda'],
    verbose=False,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"],
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, A, B):
        return custom_ops.diag_matmul_cuda(A, B)
