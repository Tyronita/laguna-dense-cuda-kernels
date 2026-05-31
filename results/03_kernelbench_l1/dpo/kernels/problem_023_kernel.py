import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = """#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void softmax_kernel(const float* input, float* output, int size, int dim) {
    extern __shared__ float shared_mem[];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        int feature_idx = idx / dim;
        int base_idx = idx % dim;
        shared_mem[threadIdx.x] = input[idx * dim + feature_idx];
        __syncthreads();
        
        float max_val = shared_mem[threadIdx.x];
        for (int i = threadIdx.x; i < dim; i += blockDim.x) {
            max_val = max_val - shared_mem[i];
        }
        shared_mem[threadIdx.x] = max_val * (1.0f - max_val + 1e-5f);
        __syncthreads();
        
        for (int i = threadIdx.x; i < dim; i += blockDim.x) {
            output[idx * dim + base_idx] = shared_mem[threadIdx.x] * shared_mem[i];
        }
    }
}

torch::Tensor softmax_cuda(torch::Tensor x) {
    auto size = x.numel();
    auto dim = x.dim();
    auto output = torch::empty_like(x);
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;
    const int shared_mem_size = block_size * dim;
    
    softmax_kernel<<<num_blocks, block_size, shared_mem_size>>>(
        x.data_ptr<float>(), output.data_ptr<float>(), size, dim
    );
    return output;
}"""

cpp_source = """torch::Tensor softmax_cuda(torch::Tensor x);"""

custom_ops = load_inline(
    name="custom_ops",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['softmax_cuda'],
    verbose=False,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"],
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x):
        return custom_ops.softmax_cuda(x)
