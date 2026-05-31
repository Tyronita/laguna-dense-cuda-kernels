import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = """#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// CUDA kernel for scaled dot product attention with manual loop unrolling
template <typename scalar_t>
__global__ void scaled_dot_product_attention_kernel_unrolled(
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ Q,
    const scalar_t* __restrict__ K,
    const scalar_t* __restrict__ V,
    const int batch_size,
    const int num_heads,
    const int sequence_length,
    const int embedding_dim) {
    
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_elements = batch_size * num_heads * sequence_length * embedding_dim;
    
    if (idx < total_elements) {
        const int row = idx / (num_heads * sequence_length);
        const int col = idx % (num_heads * sequence_length);
        const int head = col / sequence_length;
        const int seq = col % sequence_length;
        
        scalar_t sum = 0;
        #pragma unroll
        for (int k = 0; k < sequence_length; ++k) {
            const scalar_t* q_row = &Q[row * num_heads * sequence_length * embedding_dim +
                                          head * sequence_length * embedding_dim +
                                          seq * embedding_dim];
            const scalar_t* k_row = &K[row * num_heads * sequence_length * embedding_dim +
                                          head * sequence_length * embedding_dim +
                                          seq * embedding_dim];
            const scalar_t* v_row = &V[row * num_heads * sequence_length * embedding_dim +
                                          head * sequence_length * embedding_dim +
                                          seq * embedding_dim];
            
            sum += q_row[k] * k_row[k];
        }
        output[idx] = sum / sequence_length;
    }
}

// PyTorch wrapper function
torch::Tensor forward_cuda(torch::Tensor Q, torch::Tensor K, torch::Tensor V) {
    TORCH_CHECK(Q.device().is_cuda(), "Q must be a CUDA tensor");
    TORCH_CHECK(K.device().is_cuda(), "K must be a CUDA tensor");
    TORCH_CHECK(V.device().is_cuda(), "V must be a CUDA tensor");

    const int batch_size = Q.size(0);
    const int num_heads = Q.size(1);
    const int sequence_length = Q.size(2);
    const int embedding_dim = Q.size(3);

    auto options = torch::TensorOptions()
        .dtype(torch::kFloat32)
        .device(torch::kCUDA, A.get_device())
        .requires_grad(false);
    auto output = torch::empty({batch_size, num_heads, sequence_length}, options);

    const int threads = 256;
    const int blocks = (batch_size * num_heads * sequence_length * embedding_dim + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(Q.scalar_type(), "scaled_dot_product_attention_kernel_unrolled", ([&] {
        scaled_dot_product_attention_kernel_unrolled<scalar_t><<<blocks, threads>>>(
            output.data_ptr<scalar_t>(),
            Q.data_ptr<scalar_t>(),
            K.data_ptr<scalar_t>(),
            V.data_ptr<scalar_t>(),
            batch_size,
            num_heads,
            sequence_length,
            embedding_dim
        );
    }));

    return output;
}"""

cpp_source = """torch::Tensor forward_cuda(torch::Tensor Q, torch::Tensor K, torch::Tensor V);"""

custom_ops = load_inline(
    name="custom_ops",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['forward_cuda'],
    verbose=False,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"],
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, Q, K, V):
        return custom_ops.forward_cuda(Q, K, V)
