import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = """#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void maxReductionKernel(const float* input, float* output, int size, int dim) {
    extern __shared__ float sharedMem[];
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int idx = bid * dim + tid;
    if (idx < size) {
        sharedMem[tid] = input[idx];
    }
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sharedMem[tid] += sharedMem[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) {
        output[bid] = sharedMem[0];
    }
}

torch::Tensor maxReduction_cuda(torch::Tensor x, int dim) {
    auto size = x.numel();
    auto output = torch::zeros({size / dim}, x.options());
    const int threads = 256;
    const int blocks = (size + threads - 1) / threads;
    maxReductionKernel<<<blocks, threads, threads * sizeof(float)>>>(x.data_ptr<float>(), output.data_ptr<float>(), size, dim);
    return output[0].reshape(x.size(0), -1);
}

torch::Tensor elementwise_add_cuda(torch::Tensor a, torch::Tensor b) {
    auto size = a.numel();
    auto out = torch::zeros_like(a);
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;
    elementwise_add_kernel<<<num_blocks, block_size>>>(a.data_ptr<float>(), b.data_ptr<float>(), out.data_ptr<float>(), size);
    return out;
}

torch::Tensor minReduction_cuda(torch::Tensor x, int dim) {
    auto size = x.numel();
    auto output = torch::zeros({size / dim}, x.options());
    const int threads = 256;
    const int blocks = (size + threads - 1) / threads;
    maxReductionKernel<<<blocks, threads, threads * sizeof(float)>>>(x.data_ptr<float>(), output.data_ptr<float>(), size, dim);
    return output[0].reshape(x.size(0), -1);
}"""

cpp_source = """torch::Tensor maxReduction_cuda(torch::Tensor x, int dim);\ntorch::Tensor elementwise_add_cuda(torch::Tensor a, torch::Tensor b);\ntorch::Tensor minReduction_cuda(torch::Tensor x, int dim);"""

custom_ops = load_inline(
    name="custom_ops",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['maxReduction_cuda', 'elementwise_add_cuda', 'minReduction_cuda'],
    verbose=False,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"],
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x):
        return custom_ops.maxReduction_cuda(x)
