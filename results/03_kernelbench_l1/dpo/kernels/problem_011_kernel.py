import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = """#include <torch/extension.h>
#include <cuda_runtime.h>

// Custom CUDA kernel for 4D tensor-matrix multiplication using shared memory tiling
__global__ void einsum_kernel(const float* A, const float* B, float* C, int BATCH, int I, int J, int L, int K) {
    // Each block handles one (b, i, j) slice
    int b = blockIdx.x;
    int i = blockIdx.y;
    int j = blockIdx.z;
    int k = blockIdx.x;

    // Allocate shared memory for tiles of A and B
    extern __shared__ float shared_mem[];
    float* A_tile = shared_mem;
    float* B_tile = shared_mem + L * K;

    float sum = 0.0f;

    // Loop over tiles in the l dimension
    for (int t = 0; t < (L + blockDim.x - 1) / blockDim.x; ++t) {
        // Load A tile: each thread loads one element
        int a_idx = b * (I * J * L) + i * (J * L) + j * L + t * blockDim.x + threadIdx.x;
        A_tile[threadIdx.x] = (a_idx < L) ? A[a_idx] : 0.0f;

        // Load B tile: each thread loads one element
        int b_idx = (t * blockDim.x + threadIdx.x) * K + k;
        B_tile[threadIdx.x] = (b_idx < K) ? B[b_idx] : 0.0f;

        __syncthreads();

        // Compute partial dot product for this tile
        #pragma unroll
        for (int l = 0; l < blockDim.x; ++l) {
            sum += A_tile[l] * B_tile[l];
        }

        __syncthreads();
    }

    // Write the result to C
    int c_idx = b * (I * J * K) + i * (J * K) + j * K + k;
    C[c_idx] = sum;
}

// The forward function exposed via PyBind11
torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.dim() == 4, "A must be 4D");
    TORCH_CHECK(B.dim() == 2, "B must be 2D");
    TORCH_CHECK(A.size(3) == B.size(0), "A.size(3) must match B.size(0)");

    int BATCH = A.size(0), I = A.size(1), J = A.size(2), L = A.size(3);
    int K = B.size(1);

    auto C = torch::zeros({BATCH, I, J, K}, A.options());

    // Define block dimensions: each block handles one (b, i, j) slice
    const int block_size = 256;
    dim3 blocks(BATCH, I, J, K);
    dim3 threads(block_size);

    // Shared memory size: two floats per tile
    size_t shared_mem_size = 2 * block_size * sizeof(float);

    einsum_kernel<<<blocks, threads, shared_mem_size>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), BATCH, I, J, L, K
    );

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
