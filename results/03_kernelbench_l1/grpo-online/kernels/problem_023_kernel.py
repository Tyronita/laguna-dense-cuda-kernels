import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = """#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void softmax_kernel(const float* input, float* output, int size, int dim) {
    extern __shared__ float shared_mem[];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float max_val = -FLT_MAX;
        for (int i = 0; i < dim; i++) {
            max_val = fmaxf(input[idx * dim + i], max_val);
        }
        shared_mem[threadIdx.x] = max_val;
        __syncthreads();
        float sum = 0.0f;
        for (int i = 0; i < dim; i++) {
            sum += input[idx * dim + i] - shared_mem[i];
        }
        shared_mem[threadIdx.x] = sum;
        __syncthreads();
        output[idx] = sum / (dim - 1.0f);
    }
}

torch::Tensor softmax_cuda(torch::Tensor x) {
    auto size = x.numel();
    auto dim = x.dim();
    auto output = torch::empty_like(x);
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;
    const int shared_mem_size = block_size * sizeof(float);
    softmax_kernel<<<num_blocks, block_size, shared_mem_size>>>(x.data_ptr<float>(), output.data_ptr<float>(), size, dim);
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
