import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = """#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Custom CUDA kernel for elementwise addition with optimized block size
__global__ void elementwise_add_kernel(const float* a, const float* b, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = a[idx] + b[idx];
    }
}

// Custom CUDA kernel for softmax with optimized block size
__global__ void softmax_kernel(const float* input, float* output, int size, int dim) {
    extern __shared__ float shared_mem[];
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int tid_in_block = tid;
    int num_threads = blockDim.x;
    int block_size = blockDim.x * blockDim.y;
    int block_offset = bid * block_size;
    int block_end = min(block_size, size);

    // Load input into shared memory
    for (int i = tid; i < block_end; i += num_threads) {
        shared_mem[i] = input[block_offset + i];
    }
    __syncthreads();

    // Compute softmax for the block
    for (int i = 0; i < block_end; i++) {
        float max_val = shared_mem[i];
        for (int j = 0; j < dim; j++) {
            float sum = max_val - shared_mem[i + j];
            shared_mem[i + j] = sum * sum;
        }
    }
    __syncthreads();

    // Write output
    for (int i = 0; i < block_end; i++) {
        float sum = 0.0f;
        for (int j = 0; j < dim; j++) {
            sum += shared_mem[i + j];
        }
        output[block_offset + i] = sum / dim;
    }
}

// Custom CUDA function for elementwise addition
torch::Tensor elementwise_add_cuda(torch::Tensor a, torch::Tensor b) {
    auto size = a.numel();
    auto out = torch::zeros_like(a);
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;
    elementwise_add_kernel<<<num_blocks, block_size>>>(a.data_ptr<float>(), b.data_ptr<float>(), out.data_ptr<float>(), size);
    return out;
}

// Custom CUDA function for softmax
torch::Tensor softmax_cuda(torch::Tensor x) {
    auto size = x.numel();
    auto dim = x.dim();
    auto output = torch::zeros_like(x);
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;
    softmax_kernel<<<num_blocks, block_size, block_size * sizeof(float)>>>(x.data_ptr<float>(), output.data_ptr<float>(), size, dim);
    return output;
}

// C++ interface exposed to PyTorch"""

cpp_source = """torch::Tensor elementwise_add_cuda(torch::Tensor a, torch::Tensor b);\ntorch::Tensor softmax_cuda(torch::Tensor x);"""

custom_ops = load_inline(
    name="custom_ops",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['elementwise_add_cuda', 'softmax_cuda'],
    verbose=False,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"],
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x):
        return custom_ops.elementwise_add_cuda(x)
