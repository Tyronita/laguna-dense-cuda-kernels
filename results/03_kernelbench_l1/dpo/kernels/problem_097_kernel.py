import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = """#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Custom CUDA kernel for fused scaled dot product attention and softmax
// Each block processes one (n, h) slice of the input tensors.
// The inner dimension (seq_len and embed_dim) is assumed to be divisible by 4 for better memory coalescing.

__global__ void fused_scaled_dot_product_attention_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ out,
    int n, int h, int seq_len, int embed_dim) {

    // Each block corresponds to one (n, h) slice
    int slice_idx = blockIdx.x;  // slice index
    int tid = threadIdx.x;
    int stride = blockDim.x;

    // Compute the dot product for this slice
    float sum = 0.0f;
    int base_idx = slice_idx * (seq_len * embed_dim);
    int k_base = slice_idx * (seq_len * embed_dim);

    // Process the embedding dimension in chunks of 4
    for (int i = tid; i < embed_dim; i += stride) {
        #pragma unroll 4
        for (int j = 0; j < 4; j++) {
            int k_idx = base_idx + i + j * stride;
            if (i + j * stride < embed_dim) {
                // Load Q and K values
                float q_val = Q[(n * h + slice_idx) * seq_len + k_idx];
                float k_val = K[(n * h + slice_idx) * seq_len + k_idx];
                sum += q_val * k_val;
            }
        }
    }

    // Apply softmax over the sequence dimension
    for (int j = 0; j < seq_len; j++) {
        float exp_val = expf(sum - (slice_idx * (seq_len * embed_dim) + j));
        sum += exp_val;
    }

    // Write the result to output tensor
    out[slice_idx * (seq_len * embed_dim) + tid] = sum;
}

// The forward function wraps the custom CUDA kernel
// Assumes input tensors are 3D and float32, and they are on CUDA.

torch::Tensor forward(torch::Tensor Q, torch::Tensor K, torch::Tensor V) {
    TORCH_CHECK(Q.is_cuda(), "Input Q must be a CUDA tensor");
    TORCH_CHECK(K.is_cuda(), "Input K must be a CUDA tensor");
    TORCH_CHECK(V.is_cuda(), "Input V must be a CUDA tensor");
    TORCH_CHECK(Q.dtype() == torch::kFloat32, "Input Q must be float32");
    TORCH_CHECK(K.dtype() == torch::kFloat32, "Input K must be float32");
    TORCH_CHECK(V.dtype() == torch::kFloat32, "Input V must be float32");

    int n = Q.size(0);
    int h = Q.size(1);
    int seq_len = Q.size(2);
    int embed_dim = Q.size(3);

    TORCH_CHECK(K.size(0) == n, "Batch size must match");
    TORCH_CHECK(K.size(1) == h, "Number of heads must match");
    TORCH_CHECK(K.size(2) == seq_len, "Sequence length must match");
    TORCH_CHECK(K.size(3) == embed_dim, "Embedding dimension must match");
    TORCH_CHECK(V.size(0) == n, "Batch size must match");
    TORCH_CHECK(V.size(1) == h, "Number of heads must match");
    TORCH_CHECK(V.size(2) == seq_len, "Sequence length must match");
    TORCH_CHECK(V.size(3) == embed_dim, "Embedding dimension must match");

    auto out = torch::empty_like(Q);

    // Launch one block per (n, h) slice
    const int threads = 256;
    const int blocks = n * h;
    fused_scaled_dot_product_attention_kernel<<<blocks, threads>>>(
        Q.data_ptr<float>(),
        K.data_ptr<float>(),
        V.data_ptr<float>(),
        out.data_ptr<float>(),
        n, h, seq_len, embed_dim
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "Kernel failed: ", cudaGetErrorString(err));

    return out;
}"""

cpp_source = """torch::Tensor forward(torch::Tensor Q, torch::Tensor K, torch::Tensor V);"""

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

    def forward(self, Q, K, V):
        return custom_ops.forward(Q, K, V)
