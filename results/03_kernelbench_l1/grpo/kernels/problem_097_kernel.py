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

    // Each block handles one (n, h) slice
    int slice_idx = blockIdx.x;  // slice index for (n, h)
    int tid = threadIdx.x;
    int stride = blockDim.x;

    // Compute the base indices for this slice
    int base_idx = slice_idx * (seq_len * embed_dim);
    int h_offset = slice_idx * seq_len * embed_dim;

    float sum = 0.0f;

    // Process the embedding dimension in chunks of 4 elements
    int tid_stride = 4;
    for (int i = tid; i < embed_dim; i += stride) {
        // Load 4 elements at once if possible
        if (i + tid_stride < embed_dim) {
            float4 q_val = *reinterpret_cast<const float4*>(&Q[base_idx + i]);
            float4 k_val = *reinterpret_cast<const float4*>(&K[h_offset + i]);
            sum += fmaf4f(q_val.x, k_val.x, sum) + fmaf4f(q_val.y, k_val.y, sum) + fmaf4f(q_val.z, k_val.z, sum) + fmaf4f(q_val.w, k_val.w, sum);
        } else {
            for (int j = 0; j < 4 && i + j < embed_dim; j++) {
                sum += fmaf(Q[base_idx + i + j], K[h_offset + i + j]);
            }
        }
    }

    // Compute softmax over the sequence length
    float max_val = -FLT_MAX;
    for (int s = 0; s < seq_len; s++) {
        float val = fmaxf(sum + 0.1 * (base_idx + s * embed_dim + i), max_val);
        sum += val * val;
    }

    // Write the result for this slice
    for (int s = 0; s < seq_len; s++) {
        for (int j = 0; j < embed_dim; j++) {
            out[s * (n * h) + slice_idx * (seq_len * embed_dim) + j] = sum / sqrtf(1e-5f + sum);
        }
    }
}

// The forward function exposed via PyBind11
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

    auto out = torch::empty({n, h, seq_len, embed_dim}, torch::device(torch::kCUDA) + torch::kFloat32);

    // Each block processes one (n, h) slice
    int num_blocks = n * h;
    int threads = 256;
    fused_scaled_dot_product_attention_kernel<<<num_blocks, threads>>>(
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
